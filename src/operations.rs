use std::borrow::Cow;
use std::collections::HashMap;
use wgpu::{Device, ShaderModule};

pub type ShaderResources = HashMap<String, ShaderModule>;

/// Operations to be performed on the given data.
pub enum Operation {
    DOUBLE,
    ADD,
    SUBTRACT,
    MULTIPLY,
    DIVIDE,
}

pub fn decode_operation(op: &Operation) -> &str {
    match op {
        Operation::DOUBLE => "double",
        Operation::ADD => "add",
        Operation::SUBTRACT => "subtract",
        Operation::MULTIPLY => "multiply",
        Operation::DIVIDE => "divide",
    }
}

pub fn shader_name(op: &Operation, type_suffix: &str) -> String {
    let base = decode_operation(op);
    if type_suffix.is_empty() {
        base.to_string()
    } else {
        format!("{}{}", base, type_suffix)
    }
}

pub async fn add_shader_modules_from_directory(
    device: &Device,
    shaders_directory: &str,
) -> Option<ShaderResources> {
    let mut shader_module_hm = HashMap::new();

    let shader_paths = match std::fs::read_dir(shaders_directory) {
        Ok(s) => s,
        Err(_) => return None,
    }
    .map(|path| path.unwrap().path().into_os_string().into_string().unwrap())
    .collect::<Vec<String>>();

    for path in shader_paths.iter() {
        let file_name = std::path::Path::new(path)
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap();

        let shader: Cow<str> = Cow::from(
            std::fs::read_to_string(path)
                .unwrap_or_else(|_| panic!("Could not read file contents from: {}", path)),
        );
        let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(file_name),
            source: wgpu::ShaderSource::Wgsl(shader),
        });
        shader_module_hm.insert(file_name.to_owned(), cs_module);
    }

    Some(shader_module_hm)
}
