# This script generate 
# the albedo, normal, depth and motion vector
# albedo is RGB [0,1] png
# normal is RGB [0,1] png a remapping from [-1,1]
# depth is RGB [0,1] png a remapping from world space [0m,25m] (WIP completely arbitrary choice for now)
# motion vector is RGB float32 openEXR uncompressed
# for this the scene need to be preconfigured to use cycle and with the compositor nodes as seeb ub blend_config_for_add_buffers.png

base_path = 'D:/rl_dataset-gen'

import bpy
import sys
j= int(sys.argv[-1])

scene = bpy.context.scene
bpy.context.scene.render.image_settings.color_mode ='RGB'
bpy.context.scene.render.image_settings.file_format='PNG'
bpy.context.scene.render.filepath = base_path
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'CUDA'
prefs.compute_device = 'CUDA_0'
bpy.ops.wm.save_userpref()

def enable_gpus(device_type, use_cpus=False):
    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons["cycles"].preferences
    cycles_preferences.refresh_devices()
    devices = cycles_preferences.devices

    if not devices:
        raise RuntimeError("Unsupported device type")

    activated_gpus = []
    for device in devices:
        if device.type == "CPU":
            device.use = use_cpus
        else:
            device.use = True
            activated_gpus.append(device.name)
            print('activated gpu', device.name)

    cycles_preferences.compute_device_type = device_type
    bpy.context.scene.cycles.device = "GPU"

    return activated_gpus

enable_gpus("CUDA")
bpy.context.scene.use_nodes = True
bpy.context.scene.cycles.use_denoising = False
bpy.context.scene.cycles.use_preview_denoising=False

for frame in range(j, (j+1)):
  scene.frame_set(frame) 
  bpy.context.scene.cycles.device = 'GPU'
  bpy.context.scene.cycles.samples = 1
  bpy.ops.render.render(write_still=False)
      
      
