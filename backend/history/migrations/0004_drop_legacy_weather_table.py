from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('history', '0003_remove_weather_fields'),
    ]

    operations = [
        migrations.RunSQL(
            sql=[
                'DROP TABLE IF EXISTS weather_weatherlog;',
                """
                DELETE FROM auth_permission
                WHERE content_type_id IN (
                    SELECT id
                    FROM django_content_type
                    WHERE app_label = 'weather'
                );
                """,
                "DELETE FROM django_content_type WHERE app_label = 'weather';",
                "DELETE FROM django_migrations WHERE app = 'weather';",
            ],
            reverse_sql=migrations.RunSQL.noop,
        ),
    ]
