
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import os
import sqlite3
import random
import re
import glob
import numpy as np

from .display import display_rgb


class MainWindow:

    def __init__(self):
        # TK UI
        self.root = tk.Tk()
        self.root.title("OSCDeepPy")
        self.root.bind("q", self.root.quit)

class ImageView:

    def __init__(self, img=None, scale=None, roi_selection=False, initial_xy=None, initial_ij=None, root=None):
        
        self.scale = scale
        self.img_size = np.asarray([np.nan,np.nan])
        if img is not None:
            self.load_data(img)
        
        # TK UI
        if root is None:
            self.root = tk.Tk()
            self.root.title("OSCDeepPy")
            self.root.bind("q", self.root.quit)
        else:
            self.root = root
        self.canvas = tk.Canvas(self.root, cursor="tcross")
        self.canvas.config(width=self.canvas_size[1], height=self.canvas_size[0])
        self.canvas.bind("<Button-1>", self.left_click)  # left mouse button
        self.canvas.bind("<B2-Motion>", self.translate_event)  # middle mouse button
        self.canvas.bind("<Button-3>", self.right_click)  # right mouse button
        self.canvas.bind("<Button-4>", self.zoom_event)
        self.canvas.bind("<Button-5>", self.zoom_event)
        self.canvas.bind('<Motion>', self.motion)

        self.canvas.pack()
        
        self.point_map = {}
        self.points_on = False
        
        self.display_image()
        if initial_xy is not None:
            assert initial_ij is None, 'Cannot specify initial_xy and initial_ij'
            for x,y in initial_xy:
                self.add_point_at(x,y) 
        elif initial_ij is not None:
            for y,x in initial_ij:
                self.add_point_at(x,y) 
        if root is None:
            self.root.mainloop()
            
    def load_data(self, data, view='sigma', **kwargs):
        data = data.squeeze()
        if len(data.shape) == 2:
            self.mono = True
            data = np.dstack([data,data,data]) #sue me
        elif len(data.shape) == 3:
            self.mono = False
        img_size = data.shape[:2]
        if not np.allclose(img_size,self.img_size):
            self.img_size = np.asarray(img_size[:2], np.float64)
            if self.scale is None:
                self.scale = 1500 / self.img_size[1]
            self.canvas_size = self.img_size*self.scale
            self.center_cv = self.canvas_size / 2
            self.center_px = np.asarray(self.img_size[:2], np.float64) / 2
            self.img_offset = np.zeros_like(self.center_px)
        if view is None or view == 'none':
            self.pil_img = display_rgb(data, plot=False, scale=None) 
        if view == 'linear' or view == 'minmax':
            self.pil_img = display_rgb(data, plot=False, scale='linear') 
        elif view == 'sigma':
            self.pil_img = display_rgb(data, plot=False, clip_sigmas=5, scale='linear') 
        elif view == 'auto':
            self.pil_img = display_rgb(data, plot=False, scale='auto') 
        elif view == 'norm':
            self.pil_img = display_rgb(data, plot=False, scale='norm') 
        elif view == 'histogram':
            raise Exception('not implemented')
        
    def clear_roi_points(self):
        self.point_map = {}
        self.tmp_points = []
        
    def get_roi_points(self):
        return np.asarray(list(self.point_map.keys()),dtype=np.int32)
        
    def motion(self, e):
        self.mouse_x,self.mouse_y = e.x, e.y
        return "break"
        
    def zoom_event(self,e):
        if e.num == 4:
            self.scale *= 1.1
        elif e.num == 5:
            self.scale *= 0.9
        self.display_image()
        return "break" 
        
    def translate_event(self,e):
        delta = np.asarray([e.y-self.mouse_y,e.x-self.mouse_x])/self.scale
        self.img_offset += delta
        self.mouse_x,self.mouse_y = e.x, e.y
        self.display_image()
        return "break" 

    def left_click(self, e):
        if not self.points_on:
            return
        self.add_point_at(*self.cv_to_px(e.x,e.y))

    def cv_to_px(self, x_cv, y_cv): 
        _px = (np.asarray([y_cv,x_cv])-self.center_cv)/self.scale + self.center_px - self.img_offset
        return _px[1],_px[0]
        
    def px_to_cv(self, x_px, y_px): 
        _cv = (np.asarray([y_px,x_px])-self.center_px + self.img_offset)*self.scale + self.center_cv
        return _cv[1],_cv[0]

    def add_point_at(self, x_px, y_px, radius=30):
        x_cv, y_cv = self.px_to_cv(x_px, y_px)  
        key = (int(x_px), int(y_px))
        if key not in self.point_map:
            yrad,xrad = [radius*self.scale,radius*self.scale]
            rectangle_id = self.canvas.create_rectangle(x_cv-xrad, y_cv-yrad, x_cv+xrad, y_cv+yrad, outline="red")
            self.point_map[key] = rectangle_id

    def toggle_points(self, e):
        if self.points_on:
            self.tmp_points = []
            for (x, y), rectangle_id in self.point_map.items():
                self.canvas.delete(rectangle_id)
                self.tmp_points.append((x, y))
            self.point_map = {}
            self.points_on = False
        else:
            for x, y in self.tmp_points:
                self.add_point_at(x, y)
            self.points_on = True

    def right_click(self, e):
        if not self.points_on:
            return
        if len(self.point_map) == 0:
            return
        closest_point = None
        closest_sqr_distance = 0.0
        x_px, y_px = self.cv_to_px(e.x, e.y)
        for _x, _y in self.point_map.keys(): # Could optimize this if > 100s
            sqr_distance = (x_px - _x) ** 2 + (y_px - _y) ** 2
            if sqr_distance < closest_sqr_distance or closest_point is None:
                closest_point = (_x, _y)
                closest_sqr_distance = sqr_distance
        rectangle_id = self.point_map.pop(closest_point)
        self.canvas.delete(rectangle_id)

    def display_image(self):
        half_width_px = self.canvas_size/self.scale/2
        upper_px,left_px = self.center_px - self.img_offset - half_width_px
        bottom_px,right_px = self.center_px - self.img_offset + half_width_px
        
        crop = self.pil_img.crop((
            max(0,int(left_px)),
            max(0,int(upper_px)),
            min(self.img_size[1]-1,int(right_px)),
            min(self.img_size[0]-1,int(bottom_px))
        ))
        
        yoff_cv = -upper_px*self.scale if int(upper_px) < 0 else 0
        ypad_cv = (bottom_px - self.img_size[0])*self.scale if int(bottom_px) >=  self.img_size[0] else 0
        xoff_cv = -left_px*self.scale if int(left_px) < 0 else 0
        xpad_cv = (right_px - self.img_size[1])*self.scale if int(right_px) >=  self.img_size[1] else 0
        
        scale_to = (int(self.canvas_size[1] - xoff_cv - xpad_cv), int(self.canvas_size[0] - yoff_cv - ypad_cv))
        img = crop.resize(scale_to, resample=Image.Resampling.BOX)
        self.tk_img = ImageTk.PhotoImage(img, master=self.canvas)
        self.canvas.create_image(xoff_cv, yoff_cv, image=self.tk_img, anchor=tk.NW)
        self.tmp_points = []
        points = self.point_map.keys()
        for rectangle_id in self.point_map.values():
            self.canvas.delete(rectangle_id)
        self.point_map = {}
        for x, y in points:
            self.add_point_at(x, y)
