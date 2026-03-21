from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
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
        for record_data in records:
            # Force the sync status to synced
            record_data['sync_status'] = 'synced'
            
            serializer = SyncPredictionSerializer(data=record_data)
            if serializer.is_valid():
                serializer.save(user=request.user)
                synced.append(serializer.data)
        
        return Response({
            "detail": f"Successfully synced {len(synced)} records.", 
            "synced_records": synced
        }, status=status.HTTP_201_CREATED)
