from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='cv-home'),
    path('about/', views.about, name='cv-about'),
    path('box/', views.box, name='box'),
    path('box/result/<str:result_image>/', views.box_result, name='box_result'),
    path('smoothing/', views.smoothing, name='smoothing'),
    path('smoothing/result/<str:result_image>/', views.smoothing_result, name='smoothing_result'),
    path('dog/', views.dog, name='dog'),
    path('dog/result/<str:result_image>/', views.dog_result, name='dog_result'),
    path('canny/', views.canny, name='canny'),
    path('canny/result/<str:result_image>/', views.canny_result, name='canny_result'),
    path('stitch/', views.upload_and_stitch_images, name='stitch'),
    path('stitch/result/', views.stitch_result, name='stitch_result'),
    path('process-images/', views.process_images, name='process_images'), 
    path('results/', views.show_results, name='show_results'),
    path('apply-morphological/', views.apply_morphological_operation, name='apply_morphological_operation'), 
    path('morphological-results/', views.show_morphological_results, name='show_morphological_results'), 
    path('process/', views.process, name='process'),
    path('ai-human-detection/', views.ai_human_detection_view, name='ai_human_detection'),
    path('human-detection-results/', views.show_human_detection_results, name='show_human_detection_results'),
    path('human_detect/', views.ai_human_detection, name='human_detect'),

]