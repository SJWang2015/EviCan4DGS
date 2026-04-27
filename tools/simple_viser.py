import viser

server = viser.ViserServer(port=8080)

# 创建一些GUI控件
scale = server.gui.add_slider("scale", min=0.0, max=2.0, step=0.01, initial_value=1.0)
btn = server.gui.add_button("print")

@scale.on_update
def _(_event):
    print("scale =", scale.value)

@btn.on_click
def _(_event):
    print("clicked")

print("Open: http://localhost:8080")
server.sleep_forever()