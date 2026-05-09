from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('diseases', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='diseaseinfo',
            name='causes_ar',
        ),
        migrations.RemoveField(
            model_name='diseaseinfo',
            name='causes_en',
        ),
        migrations.RemoveField(
            model_name='diseaseinfo',
            name='description_ar',
        ),
        migrations.RemoveField(
            model_name='diseaseinfo',
            name='description_en',
        ),
        migrations.RemoveField(
            model_name='diseaseinfo',
            name='treatment_ar',
        ),
        migrations.RemoveField(
            model_name='diseaseinfo',
            name='treatment_en',
        ),
    ]
