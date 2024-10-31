# This script generate 
# the normal, depth and albedo map
# the motion vector
# the scene need to be preconfigured to use cycle as a blender renderer

import sys
frame_number = int(sys.argv[-1])
         
import bpy

render = bpy.context.scene.render
render.image_settings.color_mode = 'RGBA' 
render.film_transparent = True
#render.use_compositing = True #do we need this?
nodes = bpy.context.scene.node_tree.nodes
links = bpy.context.scene.node_tree.links
render_layers = nodes.new('CompositorNodeRLayers')

bpy.context.scene.render.filepath = 'd:/rl_dataset/add/'
motion_vector_file_path = bpy.context.scene.render.filepath+'Motion'

for a in render_layers.outputs.keys():
    if str(a) == "Denoising Depth" or str(a) == "Denoising Normal" or str(a) == "Denoising Albedo":
        file_output = nodes.new(type="CompositorNodeOutputFile")
        file_output.label = str(a) + " Output"
        file_output.base_path = ''
        file_output.file_slots[0].use_node_format = True
        file_output.format.file_format = 'PNG'
        file_output.format.color_mode = 'RGB'
        file_output.format.color_depth = '16'
        file_output.file_slots[0].path = bpy.context.scene.render.filepath+ str(a)
        links.new(render_layers.outputs[a], file_output.inputs[0])

    if str(a) == "Vector":
        separate_color = nodes.new(type="CompositorNodeSeparateColor")
        separate_color.label = 'Separate color'
        links.new(render_layers.outputs[a], separate_color.inputs[0])

        combine_color = nodes.new(type="CompositorNodeCombineColor")
        combine_color.label = 'Combine color'
        links.new(separate_color.outputs[0], combine_color.inputs[0])
        links.new(separate_color.outputs[1], combine_color.inputs[1])
        links.new(separate_color.outputs[2], combine_color.inputs[2])
        links.new(separate_color.outputs[3], combine_color.inputs[3])

        file_output = nodes.new(type="CompositorNodeOutputFile")
        file_output.label = 'Motion Vector Output'
        file_output.base_path = ''
        file_output.file_slots[0].use_node_format = True
        file_output.format.file_format = 'OPEN_EXR'
        file_output.format.color_mode = 'RGBA'
        file_output.format.color_depth = '32'
        file_output.file_slots[0].path = motion_vector_file_path
        links.new(combine_color.outputs[0], file_output.inputs[0])

bpy.context.scene.frame_set(frame_number)   
bpy.ops.render.render(write_still=False)