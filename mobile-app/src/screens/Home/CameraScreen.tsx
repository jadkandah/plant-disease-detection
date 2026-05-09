import React, { useState, useRef } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ActivityIndicator, Alert, Platform } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { RefreshCw, X } from 'lucide-react-native';
import apiClient from '../../services/auth/apiClient';
import { useNetworkStatus } from '../../services/network/useNetworkStatus';
import { enqueueOfflineResult } from '../../services/offline/offlineQueue';
import { predictOffline } from '../../services/offline/localInference';
import { useTranslation } from '../../store/LanguageContext';
import { useModelMode } from '../../store/ModelModeContext';

export default function CameraScreen({ navigation }: any) {
  const [facing, setFacing] = useState<'front' | 'back'>('back');
  const [permission, requestPermission] = useCameraPermissions();
  const [isProcessing, setIsProcessing] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const cameraRef = useRef<CameraView>(null);
  const { isConnected } = useNetworkStatus();
  const { t } = useTranslation();
  const { isOnlineMode } = useModelMode();

  const getUploadErrorMessage = (error: any) => {
    const backendMessage = error?.response?.data?.detail || error?.response?.data?.error || error?.message;
    const message = String(backendMessage || '');
    if (message.toLowerCase().includes('timeout') || error?.code === 'ECONNABORTED') {
      return t('gallery.requestTimeout');
    }
    return message || t('camera.uploadFailed');
  };

  if (!permission) return <View style={styles.container} />;

  if (!permission.granted) {
    return (
      <View style={styles.permissionContainer}>
        <Text style={styles.permissionText}>{t('camera.noAccess')}</Text>
        <TouchableOpacity style={styles.permissionButton} onPress={requestPermission}>
          <Text style={styles.permissionButtonText}>{t('camera.grantPermission')}</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.backLink} onPress={() => navigation.goBack()}>
          <Text style={styles.backLinkText}>{t('common.goBack')}</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const takePicture = async () => {
    if (cameraRef.current && !isProcessing) {
      try {
        setErrorMessage(null);
        setStatusMessage(t('camera.capturingImage'));
        setIsProcessing(true);
        const photo = await cameraRef.current.takePictureAsync({ quality: 0.7 });
        if (!photo) {
          setStatusMessage(null);
          setErrorMessage(t('camera.captureFailed'));
          Alert.alert(t('common.error'), t('camera.captureFailed'));
          return;
        }

        if (!isConnected || !isOnlineMode) {
          setStatusMessage(t('gallery.analyzingImage'));
          const prediction = await predictOffline(photo.uri);
          const id = `offline_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
          await enqueueOfflineResult({
            id,
            predictionKey: prediction.prediction_key,
            sourceType: 'camera',
            predictedAt: new Date().toISOString(),
            cropName: prediction.disease_info.crop_name_en,
            diseaseNameEn: prediction.disease_info.disease_name_en,
            diseaseNameAr: prediction.disease_info.disease_name_ar,
            confidence: prediction.confidence,
            isHealthy: prediction.is_healthy,
          });
          setStatusMessage(null);
          navigation.navigate('Result', { prediction });
          return;
        }

        const formData = new FormData();
        setStatusMessage(t('gallery.analyzingImage'));

        if (Platform.OS === 'web') {
          const resp = await fetch(photo.uri);
          const blob = await resp.blob();
          const file = new File([blob], 'photo.jpg', { type: blob.type || 'image/jpeg' });
          formData.append('image', file);
        } else {
          const filename = photo.uri.split('/').pop() || 'photo.jpg';
          const match = /\.(\w+)$/.exec(filename);
          const fileType = match ? `image/${match[1]}` : 'image/jpeg';
          formData.append('image', { uri: photo.uri, name: filename, type: fileType } as any);
        }

        formData.append('source_type', 'camera');
        formData.append('mode', isOnlineMode ? 'online' : 'offline');

        const response = await apiClient.post('/predict/', formData);
        setStatusMessage(null);
        navigation.navigate('Result', { prediction: response.data });
      } catch (error: any) {
        const message = getUploadErrorMessage(error);
        setStatusMessage(null);
        setErrorMessage(message);
        Alert.alert(t('common.error'), message);
      } finally {
        setIsProcessing(false);
      }
    }
  };

  const toggleCameraFacing = () => {
    setFacing((current) => (current === 'back' ? 'front' : 'back'));
  };

  return (
    <View style={styles.container}>
      <CameraView style={styles.camera} facing={facing} ref={cameraRef}>
        {(!isConnected || !isOnlineMode) && (
          <View style={styles.offlineBanner}>
            <Text style={styles.offlineText}>{t('gallery.offlineBanner')}</Text>
          </View>
        )}
        <View style={styles.buttonContainer}>
          <TouchableOpacity style={styles.iconButton} onPress={toggleCameraFacing}>
            <RefreshCw color="white" size={24} />
          </TouchableOpacity>
          <TouchableOpacity style={styles.captureButton} onPress={takePicture} disabled={isProcessing}>
            {isProcessing ? <ActivityIndicator color="#2E7D32" size="large" /> : <View style={styles.captureInner} />}
          </TouchableOpacity>
          <TouchableOpacity style={styles.iconButton} onPress={() => navigation.goBack()}>
            <X color="white" size={24} />
          </TouchableOpacity>
        </View>
        {(statusMessage || errorMessage) && (
          <View style={styles.messageBanner}>
            {statusMessage && <Text style={styles.statusText}>{statusMessage}</Text>}
            {errorMessage && <Text style={styles.errorText}>{errorMessage}</Text>}
          </View>
        )}
      </CameraView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: 'center' },
  camera: { flex: 1 },
  buttonContainer: { flex: 1, flexDirection: 'row', backgroundColor: 'transparent', justifyContent: 'space-around', alignItems: 'flex-end', marginBottom: 40 },
  iconButton: { backgroundColor: 'rgba(0,0,0,0.5)', padding: 12, borderRadius: 30, marginBottom: 20 },
  captureButton: { width: 80, height: 80, bottom: 10, borderRadius: 40, backgroundColor: 'rgba(255,255,255,0.3)', justifyContent: 'center', alignItems: 'center' },
  captureInner: { width: 60, height: 60, borderRadius: 30, backgroundColor: 'white' },
  offlineBanner: { backgroundColor: 'rgba(230, 81, 0, 0.85)', padding: 10, alignItems: 'center' },
  offlineText: { color: 'white', fontSize: 13, fontWeight: '600' },
  messageBanner: { position: 'absolute', left: 20, right: 20, bottom: 145, padding: 12, borderRadius: 8, backgroundColor: 'rgba(255,255,255,0.94)', alignItems: 'center' },
  statusText: { color: '#2E7D32', fontSize: 14, fontWeight: '600', textAlign: 'center' },
  errorText: { color: '#C62828', fontSize: 14, fontWeight: '600', textAlign: 'center' },
  permissionContainer: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 20, backgroundColor: '#f9f9f9' },
  permissionText: { fontSize: 18, color: '#333', textAlign: 'center', marginBottom: 20 },
  permissionButton: { backgroundColor: '#2E7D32', padding: 15, borderRadius: 8, marginBottom: 12 },
  permissionButtonText: { color: 'white', fontSize: 16, fontWeight: 'bold' },
  backLink: { padding: 10 },
  backLinkText: { color: '#2E7D32', fontSize: 16 },
});
