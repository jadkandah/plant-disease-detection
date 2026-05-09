from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.utils import timezone
from django.utils.dateparse import parse_datetime
from .models import PredictionRecord
from .serializers import PredictionRecordSerializer, SyncPredictionSerializer

class HistoryViewSet(viewsets.ModelViewSet):
    serializer_class = PredictionRecordSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        # Only return predictions for the logged-in user
        queryset = PredictionRecord.objects.filter(user=self.request.user).order_by('-predicted_at')
        
        # Filtering
        crop = self.request.query_params.get('crop', None)
        if crop:
            queryset = queryset.filter(crop_name__icontains=crop)
            
        date = self.request.query_params.get('date', None)
        if date:
            queryset = queryset.filter(predicted_at__date=date)
            
        return queryset

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    @action(detail=False, methods=['post'], url_path='sync')
    def sync_offline(self, request):
        """
        Receives an array of PredictionRecord payload dictionaries that 
        were created offline and missed the initial server connection.
        """
        records = request.data.get('records', [])
        if not isinstance(records, list):
            return Response({"detail": "Payload must be a list under the 'records' key."}, status=status.HTTP_400_BAD_REQUEST)

        synced = []
        errors = []
        for record_data in records:
            record_data = record_data.copy()
            predicted_at = record_data.get('predicted_at')

            # Offline/local predictions arrive after the fact; mark them as
            # synced while preserving camera/gallery as source_type.
            record_data['sync_status'] = 'synced'
            record_data['model_mode'] = 'offline'
            
            serializer = SyncPredictionSerializer(data=record_data)
            if serializer.is_valid():
                record = serializer.save(user=request.user)
                if predicted_at:
                    parsed_predicted_at = parse_datetime(predicted_at)
                    if parsed_predicted_at:
                        if timezone.is_naive(parsed_predicted_at):
                            parsed_predicted_at = timezone.make_aware(parsed_predicted_at)
                        PredictionRecord.objects.filter(pk=record.pk).update(predicted_at=parsed_predicted_at)
                        record.predicted_at = parsed_predicted_at
                synced.append(PredictionRecordSerializer(record).data)
            else:
                errors.append(serializer.errors)
        
        return Response({
            "detail": f"Successfully synced {len(synced)} records.", 
            "synced_records": synced,
            "failed_records": errors,
        }, status=status.HTTP_201_CREATED if synced else status.HTTP_400_BAD_REQUEST)
