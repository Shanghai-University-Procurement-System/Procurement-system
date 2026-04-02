from django.conf import settings
from django.urls import include, path
from django.contrib import admin
from django.urls import path
from reports.views import run_ai_agent_and_create_report
from wagtail.admin import urls as wagtailadmin_urls
from wagtail import urls as wagtail_urls
from wagtail.documents import urls as wagtaildocs_urls
from custom_auth.views import get_fastapi_token
from search import views as search_views
from reports import views as report_views

urlpatterns = [
    path("django-admin/", admin.site.urls),
    path("admin/", include(wagtailadmin_urls)),
    path("documents/", include(wagtaildocs_urls)),
    path("search/", search_views.search, name="search"),
    path("reports/ai-analysis/", report_views.ai_analysis, name="ai_analysis"),
    path("api/report/update/<int:report_id>/", report_views.update_report_content, name="update_report_content"),
    path("api/report/filter/<int:report_id>/", report_views.filter_historical_projects, name="filter_historical_projects"),
    path("api/report/filter/ongoing/<int:report_id>/", report_views.filter_ongoing_projects, name="filter_ongoing_projects"),
    path("api/report/filter/intentions/<int:report_id>/", report_views.filter_purchase_intentions, name="filter_purchase_intentions"),
    path("api/report/filter/announcements/<int:report_id>/", report_views.filter_announcements, name="filter_announcements"),
    path("api/report/filter/contracts/<int:report_id>/", report_views.filter_contracts, name="filter_contracts"),
    path("api/report/filter/documents/<int:report_id>/", report_views.filter_documents, name="filter_documents"),
    path("accounts/", include("custom_auth.urls")), # Override password reset
    path("accounts/", include("allauth.urls")),
    path('api/run-agent/', run_ai_agent_and_create_report, name='run_agent'),
    path('api/get-token/', get_fastapi_token, name='get_fastapi_token'),

]


if settings.DEBUG:
    from django.conf.urls.static import static
    from django.contrib.staticfiles.urls import staticfiles_urlpatterns

    # Serve static and media files from development server
    urlpatterns += staticfiles_urlpatterns()
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

urlpatterns = urlpatterns + [
    # For anything not caught by a more specific rule above, hand over to
    # Wagtail's page serving mechanism. This should be the last pattern in
    # the list:
    path("", include(wagtail_urls)),
    # Alternatively, if you want Wagtail pages to be served from a subpath
    # of your site, rather than the site root:
    #    path("pages/", include(wagtail_urls)),
]
