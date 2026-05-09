from django.contrib import admin
from .models import DiseaseInfo


@admin.register(DiseaseInfo)
class DiseaseInfoAdmin(admin.ModelAdmin):
    list_display = ('class_key', 'crop_name_en', 'disease_name_en', 'health_status')
    list_filter = ('health_status', 'crop_name_en')
    search_fields = ('class_key', 'crop_name_en', 'disease_name_en', 'crop_name_ar', 'disease_name_ar')
    ordering = ('crop_name_en', 'disease_name_en')
    readonly_fields = ('class_key',)

    fieldsets = (
        ('Classification', {
            'fields': ('class_key', 'health_status'),
        }),
        ('English', {
            'fields': ('crop_name_en', 'disease_name_en'),
        }),
        ('Arabic (عربي)', {
            'fields': ('crop_name_ar', 'disease_name_ar'),
        }),
    )
