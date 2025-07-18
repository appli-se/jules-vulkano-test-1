// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

// This example is a copy of the "Clear screen" example from the vulkano-examples repository,
// with minor modifications to make it work in this environment.

use std::sync::Arc;
use vulkano::{
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        RenderPassBeginInfo, SubpassBeginInfo, SubpassContents, SubpassEndInfo,
    },
    device::{
        physical::{PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo, QueueFlags,
    },
    image::{view::ImageView, ImageUsage},
    instance::{Instance, InstanceCreateInfo},
    render_pass::{Framebuffer, FramebufferCreateInfo},
    swapchain::{
        acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
    },
    sync::{self, GpuFuture},
    VulkanError,
    Validated,
    library::VulkanLibrary,
};
use winit::{
    event::{WindowEvent},
    event_loop::{EventLoop},
    window::Window,
};

use winit::application::ApplicationHandler;
use winit::event_loop::ActiveEventLoop;

use vulkano::device::Queue;
use vulkano::image::Image;

struct App {
    window: Option<Arc<Window>>,
    device: Option<Arc<Device>>,
    queue: Option<Arc<Queue>>,
    swapchain: Option<Arc<Swapchain>>,
    images: Option<Vec<Arc<Image>>>,
    command_buffer_allocator: Option<Arc<StandardCommandBufferAllocator>>,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
    recreate_swapchain: bool,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let mut window_attributes = winit::window::WindowAttributes::default();
        window_attributes.title = "Vulkano Test".to_string();
        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
        self.window = Some(window.clone());

        // Create the Vulkan instance
        let library = VulkanLibrary::new().unwrap();
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                enabled_extensions: Surface::required_extensions(event_loop).unwrap(),
                ..Default::default()
            },
        )
        .unwrap();

        // Create the Vulkan surface
        let surface =
            Surface::from_window(instance.clone(), window.clone()).unwrap();

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..Default::default()
        };
        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .find(|(i, q)| {
                        q.queue_flags.contains(QueueFlags::GRAPHICS)
                            && p.surface_support(*i as u32, &surface).unwrap_or(false)
                    })
                    .map(|(i, _)| (p.clone(), i as u32))
            })
            .min_by_key(|(p, _)| {
                // We assign a lower score to device types that are likely to be faster/better.
                match p.properties().device_type {
                    PhysicalDeviceType::DiscreteGpu => 0,
                    PhysicalDeviceType::IntegratedGpu => 1,
                    PhysicalDeviceType::VirtualGpu => 2,
                    PhysicalDeviceType::Cpu => 3,
                    PhysicalDeviceType::Other => 4,
                    _ => 5,
                }
            })
            .expect("no suitable physical device found");

        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .unwrap();

        self.device = Some(device.clone());
        self.queue = Some(queues.next().unwrap());

        let (swapchain, images) = {
            let surface_capabilities = device
                .physical_device()
                .surface_capabilities(&surface, Default::default())
                .unwrap();
            let image_format = device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .unwrap()[0]
                .0;

            let (swapchain, images) = Swapchain::new(
                device.clone(),
                surface.clone(),
                SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count,
                    image_format,
                    image_extent: window.inner_size().into(),
                    image_usage: ImageUsage::COLOR_ATTACHMENT,
                    composite_alpha: surface_capabilities
                        .supported_composite_alpha
                        .into_iter()
                        .next()
                        .unwrap(),
                    ..Default::default()
                },
            )
            .unwrap();
            (swapchain, images)
        };
        self.swapchain = Some(swapchain);
        self.images = Some(images);

        self.command_buffer_allocator = Some(Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        )));
        self.previous_frame_end = Some(sync::now(device.clone()).boxed());
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                self.recreate_swapchain = true;
            }
            WindowEvent::RedrawRequested => {
                let image_extent: [u32; 2] = self.window.as_ref().unwrap().inner_size().into();
                if image_extent.contains(&0) {
                    return;
                }

                self.previous_frame_end.as_mut().unwrap().cleanup_finished();

                if self.recreate_swapchain {
                    let (new_swapchain, new_images) = self.swapchain.as_ref().unwrap()
                        .recreate(SwapchainCreateInfo {
                            image_extent,
                            ..self.swapchain.as_ref().unwrap().create_info()
                        })
                        .expect("failed to recreate swapchain");

                    self.swapchain = Some(new_swapchain);
                    self.images = Some(new_images);
                    self.recreate_swapchain = false;
                }

                let (image_index, suboptimal, acquire_future) =
                    match acquire_next_image(self.swapchain.as_ref().unwrap().clone(), None) {
                        Ok(r) => r,
                        Err(Validated::Error(VulkanError::OutOfDate)) => {
                            self.recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("failed to acquire next image: {e}"),
                    };

                if suboptimal {
                    self.recreate_swapchain = true;
                }

                let clear_values = vec![Some([0.0, 0.0, 1.0, 1.0].into())];

                let mut builder = AutoCommandBufferBuilder::primary(
                    self.command_buffer_allocator.as_ref().unwrap().clone(),
                    self.queue.as_ref().unwrap().queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                let render_pass = vulkano::single_pass_renderpass!(
                    self.device.as_ref().unwrap().clone(),
                    attachments: {
                        color: {
                            format: self.swapchain.as_ref().unwrap().image_format(),
                            samples: 1,
                            load_op: Clear,
                            store_op: Store,
                        }
                    },
                    pass: {
                        color: [color],
                        depth_stencil: {}
                    }
                )
                .unwrap();

                let framebuffers = self.images.as_ref().unwrap()
                    .iter()
                    .map(|image| {
                        let view = ImageView::new_default(image.clone()).unwrap();
                        Framebuffer::new(
                            render_pass.clone(),
                            FramebufferCreateInfo {
                                attachments: vec![view],
                                ..Default::default()
                            },
                        )
                        .unwrap()
                    })
                    .collect::<Vec<_>>();

                builder
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values,
                            ..RenderPassBeginInfo::framebuffer(
                                framebuffers[image_index as usize].clone(),
                            )
                        },
                        SubpassBeginInfo {
                            contents: SubpassContents::Inline,
                            ..Default::default()
                        },
                    )
                    .unwrap()
                    .end_render_pass(SubpassEndInfo::default())
                    .unwrap();

                let command_buffer = builder.build().unwrap();

                let future = self.previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(self.queue.as_ref().unwrap().clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(
                        self.queue.as_ref().unwrap().clone(),
                        SwapchainPresentInfo::swapchain_image_index(self.swapchain.as_ref().unwrap().clone(), image_index),
                    )
                    .then_signal_fence_and_flush();

                match future {
                    Ok(future) => {
                        self.previous_frame_end = Some(future.boxed());
                    }
                    Err(Validated::Error(VulkanError::OutOfDate)) => {
                        self.recreate_swapchain = true;
                        self.previous_frame_end = Some(sync::now(self.device.as_ref().unwrap().clone()).boxed());
                    }
                    Err(e) => {
                        println!("failed to flush future: {e}");
                        self.previous_frame_end = Some(sync::now(self.device.as_ref().unwrap().clone()).boxed());
                    }
                }
            }
            _ => (),
        }
    }
}


fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App {
        window: None,
        device: None,
        queue: None,
        swapchain: None,
        images: None,
        command_buffer_allocator: None,
        previous_frame_end: None,
        recreate_swapchain: false,
    };
    event_loop.run_app(&mut app).unwrap();
}
