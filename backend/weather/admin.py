from django.contrib import admin
from .models import WeatherLog


@admin.register(WeatherLog)
class WeatherLogAdmin(admin.ModelAdmin):
    list_display = ('city_name', 'temperature', 'humidity', 'risk_level', 'user', 'fetched_at')
    list_filter = ('risk_level', 'city_name')
    search_fields = ('city_name', 'user__email')
    ordering = ('-fetched_at',)
    date_hierarchy = 'fetched_at'
    readonly_fields = ('fetched_at',)

    fieldsets = (
        ('Location', {
            'fields': ('user', 'latitude', 'longitude', 'city_name', 'country'),
        }),
        ('Weather Conditions', {
            'fields': ('temperature', 'humidity', 'wind_speed', 'feels_like',
                       'pressure', 'weather_code', 'description'),
        }),
        ('Disease Risk', {
            'fields': ('risk_level', 'risk_message'),
        }),
        ('Metadata', {
            'fields': ('fetched_at',),
        }),
    )
