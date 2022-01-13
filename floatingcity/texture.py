import numpy as np
import cairo
from PIL import Image

def create_window_texture():

    # generate the texture
    size = (512, 512)
    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, *size)
    ctx = cairo.Context(surface)
    ctx.scale(*size)


    ctx.set_source_rgb(1.0,1.0,1.0)
    num_windows_x = 9
    num_windows_y = 15

    window_spacing_x = 1.0 / num_windows_x
    window_size_x = 0.5 / num_windows_x
    window_spacing_y = 1.0 / num_windows_y
    window_size_y = 0.5 / num_windows_y


    for x in np.arange(window_size_x/2, 1.0, window_spacing_x):
        for y in np.arange(window_size_y/2, 1.0, window_spacing_y):
            ctx.rectangle(x,y,window_size_x,window_size_y)
            ctx.fill()

    surface.write_to_png('output/textures/building_texture.png')

    window_texture = Image.open('output/textures/building_texture.png')
    return window_texture

def create_path_texture():

    # generate the texture
    sizex, sizey = (512, 512)
    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, sizex, sizey)
    ctx = cairo.Context(surface)

    ctx.set_source_rgb(1.0,1.0,1.0)
    number_of_stripes = 20
    strip_width = 4

    for y in np.arange(0, sizey, int(sizey / number_of_stripes)):
        ctx.rectangle(y, 0, strip_width, sizex)
        ctx.fill()

    surface.write_to_png('output/textures/path_texture.png')

    window_texture = Image.open('output/textures/path_texture.png')
    return window_texture





