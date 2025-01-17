import gradio as gr
import plotly.express as px
from theme_classifier import ThemeClassifier

def get_themes(theme_list, subtitles_path, save_path):
    # Split theme list and initialize classifier
    theme_list = theme_list.split(',')
    theme_classifier = ThemeClassifier(theme_list)
    output_df = theme_classifier.get_themes(subtitles_path, save_path)
    
    # Remove 'dialogue' from the theme list
    if 'dialogue' in theme_list:
        theme_list.remove('dialogue')
    
    # Summarize scores
    output_df = output_df[theme_list].sum().reset_index()
    output_df.columns = ['Theme', 'Score']
    
    # Create Plotly bar plot
    fig = px.bar(
        output_df, 
        x='Score', 
        y='Theme', 
        orientation='h', 
        title="Series Theme",
        labels={'Score': 'Score', 'Theme': 'Theme'},
        text='Score'
    )
    
    return fig

def main():
    with gr.Blocks() as iface:
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Theme Classification (Zero Shot Classifiers)</h1>")
                with gr.Row():
                    with gr.Column():
                        plot = gr.Plot()  # Correct Gradio component for plots
                    with gr.Column():
                        theme_list = gr.Textbox(label="Themes")
                        subtitles_path = gr.Textbox(label="Subtitles or Script Path")
                        save_path = gr.Textbox(label="Save Path")
                        get_themes_button = gr.Button("Get Themes")
                        
                        # Connect button to the function
                        get_themes_button.click(
                            get_themes, 
                            inputs=[theme_list, subtitles_path, save_path], 
                            outputs=[plot]
                        )
                    
    iface.launch(share=True)
    
if __name__ == '__main__':
    main()
