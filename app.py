# Credits to Code Institute: All code below is either; 
# re-modeled, re-structed or re-created to fit this project.

from app_pages.multipage import MultiPage

from app_pages.page_summary import page_summary_body
from app_pages.page_mildew_visualizer import page_mildew_visualizer_body
from app_pages.page_mildew_detector import page_mildew_detector_body
from app_pages.page_project_hypothesis import page_project_hypothesis_body
from app_pages.page_ml_performance import page_ml_performance_metrics

app = MultiPage(app_name="Mildew Detection in Cherry Leaves")

app.add_page("Quick Project Summary", page_summary_body)
app.add_page("Cherry Leaf Mildew Visualiser", page_mildew_visualizer_body)
app.add_page("Cherry Leaf Mildew Detection", page_mildew_detector_body)
app.add_page("Project Hypothesis", page_project_hypothesis_body)
app.add_page("ML Performance Metrics", page_ml_performance_metrics)

app.run()