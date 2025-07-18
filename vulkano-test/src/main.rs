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
    },
    device::{
        physical::{PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo, QueueFlags,
    },
    image::{Image, ImageUsage},
    instance::{Instance, InstanceCreateInfo},
    library::VulkanLibrary,
    format::Format,
    image::ImageCreateInfo,
    memory::allocator::{StandardMemoryAllocator, AllocationCreateInfo, MemoryTypeFilter},
    sync::GpuFuture,
};
fn main() {
    // Create the Vulkan instance
    let library = VulkanLibrary::new().unwrap();
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions: vulkano::instance::InstanceExtensions {
                khr_surface: true,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();

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
                .find(|(_i, q)| {
                    q.queue_flags.contains(QueueFlags::GRAPHICS)
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

    let queue = queues.next().unwrap();
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    let image = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            format: Format::R8G8B8A8_UNORM,
            extent: [1024, 768, 1],
            usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        }
    ).unwrap();

    let command_buffer_allocator =
        Arc::new(StandardCommandBufferAllocator::new(device.clone(), Default::default()));

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
                format: image.format(),
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

    let view = vulkano::image::view::ImageView::new_default(image.clone()).unwrap();
    let framebuffer = vulkano::render_pass::Framebuffer::new(
        render_pass.clone(),
        vulkano::render_pass::FramebufferCreateInfo {
            attachments: vec![view],
            ..Default::default()
        },
    )
    .unwrap();

    builder
        .begin_render_pass(
            vulkano::command_buffer::RenderPassBeginInfo {
                clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],
                ..vulkano::command_buffer::RenderPassBeginInfo::framebuffer(framebuffer.clone())
            },
            vulkano::command_buffer::SubpassBeginInfo {
                contents: vulkano::command_buffer::SubpassContents::Inline,
                ..Default::default()
            },
        )
        .unwrap()
        .end_render_pass(vulkano::command_buffer::SubpassEndInfo::default())
        .unwrap();

    let command_buffer = builder.build().unwrap();

    let future = vulkano::sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();
}
