# Credits to Code Institute: All code below is either; 
# re-modeled, re-structed or re-created to fit this project.

import streamlit as st

class MultiPage:
        
    def __init__(self, app_name) -> None:
        self.pages = []
        self.app_name = app_name

    def add_page(self, title, func) -> None:
        self.pages.append({"title": title, "function": func})

    def run(self):
        st.title(self.app_name)
        page = st.sidebar.radio('Menu', self.pages,
                                format_func=lambda page: page['title'])
        page['function']()