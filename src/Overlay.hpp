#pragma once
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/extensions/shape.h>
#include <X11/extensions/Xfixes.h>
#include <iostream>

class VisualDebugger {
    Display* display;
    Window overlay;
    GC gc;
    int screen_width, screen_height;

public:
    VisualDebugger() {
        display = XOpenDisplay(NULL);
        int screen = DefaultScreen(display);
        screen_width = DisplayWidth(display, screen);
        screen_height = DisplayHeight(display, screen);

        XSetWindowAttributes attrs;
        attrs.override_redirect = True;
        attrs.background_pixel = 0;

        overlay = XCreateWindow(display, DefaultRootWindow(display), 
                               0, 0, screen_width, screen_height, 0, 
                               DefaultDepth(display, screen), InputOutput, 
                               DefaultVisual(display, screen), 
                               CWOverrideRedirect | CWBackPixel, &attrs);

        // Make window input-transparent (click-through)
        XRectangle rect;
        XserverRegion region = XFixesCreateRegion(display, &rect, 0);
        XFixesSetWindowShapeRegion(display, overlay, ShapeInput, 0, 0, region);
        XFixesDestroyRegion(display, region);

        XMapWindow(display, overlay);
        XLowerWindow(display, overlay); // Keep it behind main app but in front of game if needed

        gc = XCreateGC(display, overlay, 0, NULL);
        XSetForeground(display, gc, 0x00FF00); // Green for prediction
    }

    ~VisualDebugger() {
        XFreeGC(display, gc);
        XDestroyWindow(display, overlay);
        XCloseDisplay(display);
    }

    void draw_prediction(int x, int y) {
        XClearWindow(display, overlay);
        // Draw crosshair at prediction point
        XDrawLine(display, overlay, gc, x - 10, y, x + 10, y);
        XDrawLine(display, overlay, gc, x, y - 10, x, y + 10);
        XFlush(display);
    }
};
