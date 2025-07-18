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
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

fn main() {
    // Create the window
    let event_loop = EventLoop::new().unwrap();
    let mut window_attributes = winit::window::WindowAttributes::default();
    window_attributes.title = "Vulkano Test".to_string();
    let window = Arc::new(Window::new(&event_loop, window_attributes).unwrap());

    // Create the Vulkan instance
    let library = VulkanLibrary::new().unwrap();
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions: Surface::required_extensions(&event_loop).unwrap(),
            ..Default::default()
        },
    )
    .unwrap();

    // Create the Vulkan surface
    let surface = unsafe {
        Surface::from_window(
            instance.clone(),
            window.clone()
        ).unwrap()
    };

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
                .position(|(i, q)| {
                    q.queue_flags.contains(QueueFlags::GRAPHICS)
                        && p.surface_support(i as u32, &surface).unwrap_or(false)
                })
                .map(|i| (p, i as u32))
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

    let queue = queues.next().unwrap();

    let (mut swapchain, mut images) = {
        let surface_capabilities = device
            .physical_device()
            .surface_capabilities(&surface, Default::default())
            .unwrap();
        let image_format = device
            .physical_device()
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0;

        Swapchain::new(
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
        .unwrap()
    };

    let command_buffer_allocator =
        Arc::new(StandardCommandBufferAllocator::new(device.clone(), Default::default()));
    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                elwt.exit();
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                recreate_swapchain = true;
            }
            Event::WindowEvent {
                event: WindowEvent::RedrawRequested,
                ..
            } => {
                let image_extent: [u32; 2] = window.inner_size().into();
                if image_extent.contains(&0) {
                    return;
                }

                previous_frame_end.as_mut().unwrap().cleanup_finished();

                if recreate_swapchain {
                    let (new_swapchain, new_images) = swapchain
                        .recreate(SwapchainCreateInfo {
                            image_extent,
                            ..swapchain.create_info()
                        })
                        .expect("failed to recreate swapchain");

                    swapchain = new_swapchain;
                    images = new_images;
                    recreate_swapchain = false;
                }

                let (image_index, suboptimal, acquire_future) =
                    match acquire_next_image(swapchain.clone(), None) {
                        Ok(r) => r,
                        Err(Validated::Error(VulkanError::OutOfDate)) => {
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("failed to acquire next image: {e}"),
                    };

                if suboptimal {
                    recreate_swapchain = true;
                }

                let clear_values = vec![Some([0.0, 0.0, 1.0, 1.0].into())];

                let mut builder = AutoCommandBufferBuilder::primary(
                    command_buffer_allocator.clone(),
                    queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                let render_pass = vulkano::single_pass_renderpass!(
                    device.clone(),
                    attachments: {
                        color: {
                            format: swapchain.image_format(),
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

                let framebuffers = images
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

                let future = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(
                        queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_index),
                    )
                    .then_signal_fence_and_flush();

                match future {
                    Ok(future) => {
                        previous_frame_end = Some(future.boxed());
                    }
                    Err(Validated::Error(VulkanError::OutOfDate)) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                    Err(e) => {
                        println!("failed to flush future: {e}");
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                }
            }
            _ => (),
        }
    });
}
