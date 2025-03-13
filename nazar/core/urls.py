from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    path("start-detection/", views.start_detection, name="start-detection"),
    path("detected-images/", views.display_detected_images, name="detected-images"),
    path("",views.home, name="homepage"),
    path("register/", views.register, name="register"),
    path('login/', auth_views.LoginView.as_view(template_name='login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='login'), name='logout'),
    path("live-feed/", views.live_page, name="live_page"),
    path("video-feed/", views.live_feed, name="live_feed"),
    path("aboutus/", views.aboutus, name="about"),
    path("contact/", views.contact, name="contact"),
    path("technology/", views.technology, name="technology"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

