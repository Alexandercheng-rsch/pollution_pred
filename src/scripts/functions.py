from matplotlib.colors import LinearSegmentedColormap, Normalize


color_legends = None
mapbox_token = ""

def cmap_continuous(cmap_list: list):
    _min = cmap_list[0][0]
    _max = cmap_list[-1][0]
    cmap_colors = [((loading-_min)/(_max-_min), color) for (loading, color) in cmap_list]
    cmap = LinearSegmentedColormap.from_list('name', cmap_colors)
    norm = Normalize(_min, _max)
    return cmap, norm

def add_color_legend(
        name: str,
        cmap,
        norm,
        ticks: list,
        tick_text_color: str = "black",
        text_offset: int = 0,
        top: str = "50%"
):
    global color_legends
    if color_legends is None:
        color_legends = {}
    color_legends[name] = {
        "cmap": cmap,
        "norm": norm,
        "ticks": ticks,
        "tick_text_color": tick_text_color,
        "text_offset": text_offset,
        "top": top,
    }

def status_color(val):
    color = "green" if val.lower() == "online" else "red"
    return f"color: {color}; font-weight: bold;"