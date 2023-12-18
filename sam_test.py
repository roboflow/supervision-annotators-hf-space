import gradio as gr

with gr.Blocks() as demo:
    with gr.Row():
        input_img = gr.Image(label="Input")
        output_img = gr.Image(label="Selected Segment")

    def click_handler(image, event: gr.SelectData):
        print("Click event:", event.value)
        if event is not None:
            x, y = event.value[0], event.value[1]
            print(f"Clicked at ({x}, {y})")
            if x is not None and y is not None:
                return (x, y)
            return ("", "")
        return ("", "")

    input_img.select(click_handler, [input_img])

if __name__ == "__main__":
    demo.launch(debug=True)
