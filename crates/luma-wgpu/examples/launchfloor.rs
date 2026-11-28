//! W3-LAUNCHFLOOR part A: raw wgpu dependent-launch cost split into phases (no luma on top).
//! `cargo run --release -p luma-hal-wgpu --example launchfloor [-- <idle_ms>]`
use std::time::{Duration, Instant};

const SRC: &str = "@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x < 1u) { y[0] = x[0] * 1.0001 + 1e-6; } }";

fn med(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}
fn us(d: Duration) -> f64 {
    d.as_secs_f64() * 1e6
}
fn line(name: &str, v: &mut Vec<f64>) {
    let n = v.len();
    let (p10, m, p90, mx) = (v[n / 10], v[n / 2], v[n * 9 / 10], v[n - 1]);
    println!("  {name:<28} med {m:8.2}  p10 {p10:8.2}  p90 {p90:8.2}  max {mx:9.2}");
}

fn main() {
    let idle_ms: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })).unwrap();
    let info = adapter.get_info();
    println!("adapter {} driver {} {} backend {:?}", info.name, info.driver, info.driver_info, info.backend);
    let ts = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: adapter.features() & wgpu::Features::TIMESTAMP_QUERY,
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .unwrap();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(SRC.into()) });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[0, 1].map(|b| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: b == 0 }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        }),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let mk = || device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 4096, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let (ba, bb) = (mk(), mk());
    let bg = |x: &wgpu::Buffer, y: &wgpu::Buffer| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: x.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: y.as_entire_binding() }],
        })
    };
    let groups = [bg(&ba, &bb), bg(&bb, &ba)];
    let wait = |idx: Option<wgpu::SubmissionIndex>| {
        device.poll(wgpu::PollType::Wait { submission_index: idx, timeout: None }).unwrap();
    };
    let nap = || {
        if idle_ms > 0 {
            std::thread::sleep(Duration::from_millis(idle_ms));
        }
    };
    let rounds = 300usize;
    // warm
    for _ in 0..50 {
        let mut e = device.create_command_encoder(&Default::default());
        {
            let mut p = e.begin_compute_pass(&Default::default());
            p.set_pipeline(&pipeline);
            p.set_bind_group(0, &groups[0], &[]);
            p.dispatch_workgroups(1, 1, 1);
        }
        queue.submit(Some(e.finish()));
    }
    wait(None);
    println!("idle sleep before each sample: {idle_ms} ms");

    // 1. One launch, full phases, wait(Wait) each time.
    {
        let (mut enc, mut fin, mut sub, mut wt, mut tot) = (vec![], vec![], vec![], vec![], vec![]);
        for i in 0..rounds {
            nap();
            let t0 = Instant::now();
            let mut e = device.create_command_encoder(&Default::default());
            {
                let mut p = e.begin_compute_pass(&Default::default());
                p.set_pipeline(&pipeline);
                p.set_bind_group(0, &groups[i & 1], &[]);
                p.dispatch_workgroups(1, 1, 1);
            }
            let t1 = Instant::now();
            let cb = e.finish();
            let t2 = Instant::now();
            let idx = queue.submit(Some(cb));
            let t3 = Instant::now();
            wait(Some(idx));
            let t4 = Instant::now();
            enc.push(us(t1 - t0));
            fin.push(us(t2 - t1));
            sub.push(us(t3 - t2));
            wt.push(us(t4 - t3));
            tot.push(us(t4 - t0));
        }
        println!("[1] one launch, submit + poll(Wait) each (us):");
        for (n, v) in [("encode", &mut enc), ("finish", &mut fin), ("submit", &mut sub), ("wait", &mut wt), ("total", &mut tot)] {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            line(n, v);
        }
    }
    // 2. Same, but spin on poll(Poll) until queue empty.
    {
        let (mut wt, mut tot) = (vec![], vec![]);
        for i in 0..rounds {
            nap();
            let t0 = Instant::now();
            let mut e = device.create_command_encoder(&Default::default());
            {
                let mut p = e.begin_compute_pass(&Default::default());
                p.set_pipeline(&pipeline);
                p.set_bind_group(0, &groups[i & 1], &[]);
                p.dispatch_workgroups(1, 1, 1);
            }
            queue.submit(Some(e.finish()));
            let t3 = Instant::now();
            loop {
                if device.poll(wgpu::PollType::Poll).map(|s| s.is_queue_empty()).unwrap_or(true) {
                    break;
                }
            }
            let t4 = Instant::now();
            wt.push(us(t4 - t3));
            tot.push(us(t4 - t0));
        }
        println!("[2] one launch, spin poll(Poll) (us):");
        wt.sort_by(|a, b| a.partial_cmp(b).unwrap());
        tot.sort_by(|a, b| a.partial_cmp(b).unwrap());
        line("spin", &mut wt);
        line("total", &mut tot);
    }
    // 3. N dependent launches, one submit each, one wait at the end: host enqueue cost and throughput per launch.
    for &n in &[64usize, 256] {
        let (mut per, mut host) = (vec![], vec![]);
        for _ in 0..40 {
            nap();
            let t0 = Instant::now();
            let mut last = None;
            for i in 0..n {
                let mut e = device.create_command_encoder(&Default::default());
                {
                    let mut p = e.begin_compute_pass(&Default::default());
                    p.set_pipeline(&pipeline);
                    p.set_bind_group(0, &groups[i & 1], &[]);
                    p.dispatch_workgroups(1, 1, 1);
                }
                last = Some(queue.submit(Some(e.finish())));
            }
            let t1 = Instant::now();
            wait(last);
            let t2 = Instant::now();
            host.push(us(t1 - t0) / n as f64);
            per.push(us(t2 - t0) / n as f64);
        }
        println!("[3] {n} dependent launches, submit each, wait at end (us per launch):");
        host.sort_by(|a, b| a.partial_cmp(b).unwrap());
        per.sort_by(|a, b| a.partial_cmp(b).unwrap());
        line("host (enqueue only)", &mut host);
        line("wall incl. final wait", &mut per);
    }
    // 4. N dependent launches in ONE encoder/submit (one pass each).
    for &n in &[64usize, 256] {
        let (mut per, mut host) = (vec![], vec![]);
        for _ in 0..40 {
            nap();
            let t0 = Instant::now();
            let mut e = device.create_command_encoder(&Default::default());
            for i in 0..n {
                let mut p = e.begin_compute_pass(&Default::default());
                p.set_pipeline(&pipeline);
                p.set_bind_group(0, &groups[i & 1], &[]);
                p.dispatch_workgroups(1, 1, 1);
            }
            let idx = queue.submit(Some(e.finish()));
            let t1 = Instant::now();
            wait(Some(idx));
            let t2 = Instant::now();
            host.push(us(t1 - t0) / n as f64);
            per.push(us(t2 - t0) / n as f64);
        }
        println!("[4] {n} dependent passes, one submit (us per launch):");
        host.sort_by(|a, b| a.partial_cmp(b).unwrap());
        per.sort_by(|a, b| a.partial_cmp(b).unwrap());
        line("host (encode+submit)", &mut host);
        line("wall incl. wait", &mut per);
    }
    // 5. GPU timestamp gap between consecutive dependent passes (one submit per launch, queued ahead, then wait).
    if ts {
        let period = f64::from(queue.get_timestamp_period());
        for &(n, per_submit) in &[(256usize, true), (256usize, false)] {
            let qs = device.create_query_set(&wgpu::QuerySetDescriptor { label: None, ty: wgpu::QueryType::Timestamp, count: (2 * n) as u32 });
            let resolve = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (16 * n) as u64, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
            let stage = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (16 * n) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            let (mut gaps, mut durs) = (vec![], vec![]);
            for _ in 0..10 {
                nap();
                let mut e = device.create_command_encoder(&Default::default());
                let mut last = None;
                for i in 0..n {
                    {
                        let mut p = e.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: None,
                            timestamp_writes: Some(wgpu::ComputePassTimestampWrites { query_set: &qs, beginning_of_pass_write_index: Some(2 * i as u32), end_of_pass_write_index: Some(2 * i as u32 + 1) }),
                        });
                        p.set_pipeline(&pipeline);
                        p.set_bind_group(0, &groups[i & 1], &[]);
                        p.dispatch_workgroups(1, 1, 1);
                    }
                    if per_submit {
                        let c = std::mem::replace(&mut e, device.create_command_encoder(&Default::default()));
                        last = Some(queue.submit(Some(c.finish())));
                    }
                }
                e.resolve_query_set(&qs, 0..(2 * n) as u32, &resolve, 0);
                e.copy_buffer_to_buffer(&resolve, 0, &stage, 0, (16 * n) as u64);
                let idx = queue.submit(Some(e.finish()));
                let _ = last;
                wait(Some(idx));
                let sl = stage.slice(..);
                sl.map_async(wgpu::MapMode::Read, |r| r.unwrap());
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                let data = sl.get_mapped_range().unwrap();
                let t: Vec<u64> = data.chunks(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
                drop(data);
                stage.unmap();
                for i in 0..n {
                    durs.push((t[2 * i + 1] - t[2 * i]) as f64 * period / 1e3);
                    if i + 1 < n {
                        gaps.push((t[2 * i + 2] - t[2 * i + 1]) as f64 * period / 1e3);
                    }
                }
            }
            println!("[5] {n} dependent passes, GPU timestamps, {} (us):", if per_submit { "one submit per pass (cmd bufs queued ahead)" } else { "one submit" });
            gaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
            durs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            line("gap end[i]->begin[i+1]", &mut gaps);
            line("pass begin->end", &mut durs);
        }
    }
    let _ = med;
}
