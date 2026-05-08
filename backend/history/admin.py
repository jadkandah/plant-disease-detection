from django.contrib import admin
from .models import PredictionRecord


@admin.register(PredictionRecord)
class PredictionRecordAdmin(admin.ModelAdmin):
    list_display = (
        'user', 'crop_name', 'disease_name_en', 'confidence',
        'is_healthy', 'source_type', 'sync_status', 'predicted_at',
    )
    list_filter = ('is_healthy', 'source_type', 'sync_status', 'crop_name')
    search_fields = ('user__email', 'crop_name', 'disease_name_en', 'disease_name_ar')
    ordering = ('-predicted_at',)
    readonly_fields = ('predicted_at',)
    date_hierarchy = 'predicted_at'

    fieldsets = (
        ('Prediction', {
            'fields': ('user', 'image', 'crop_name', 'disease_name_en', 'disease_name_ar',
                       'confidence', 'is_healthy', 'predicted_at'),
        }),
        ('Source & Sync', {
            'fields': ('source_type', 'sync_status'),
        }),
    )
