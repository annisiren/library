# Start with some imports!

from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets

# Very basic function
def f(x):
    return x

# Generate a slider to interact with
interact(f, x=10,);

# Booleans generate check-boxes
interact(f, x=True);
