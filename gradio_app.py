import gradio as gr
import plotly.express as px
from theme_classifier import ThemeClassifier
from character_network import NamedEntityRecognizer, CharacterNetworkGenerator

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

def get_character_network(subtitles_path, ner_path):
    ner = NamedEntityRecognizer()
    ner_df = ner.get_ners(subtitles_path, ner_path)
    
    character_network_generator = CharacterNetworkGenerator()
    relationship_df = character_network_generator.generate_character_network(ner_df)
    html = character_network_generator.draw_network_graph(relationship_df)
    
    return html

def main():
    
    # Section containing theme classification
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
                            outputs=[plot])
        # Section containing character network 
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Character Network (NERs and Graphs)</h1>")
                with gr.Row():
                    with gr.Column():
                        network_html = gr.HTML()
                    with gr.Column():
                        subtitles_path = gr.Textbox(label="Subtitles or Script Path")
                        ner_path = gr.Textbox(label="Path to Save NER Results")
                        get_network_graph_button = gr.Button("Get Character Network")
                        
                        # Connect button to the function
                        get_network_graph_button.click(
                            get_character_network, 
                            inputs=[subtitles_path, ner_path], 
                            outputs=[network_html]
                        )
                    
    iface.launch(share=True)
    
if __name__ == '__main__':
    main()
