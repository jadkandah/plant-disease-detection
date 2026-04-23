import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Image, ActivityIndicator, Alert, Platform } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import * as ImagePicker from 'expo-image-picker';
import { ArrowLeft } from 'lucide-react-native';
import apiClient from '../../services/auth/apiClient';
import { useNetworkStatus } from '../../services/network/useNetworkStatus';
import { enqueueOfflinePrediction } from '../../services/offline/offlineQueue';
import { useTranslation } from '../../store/LanguageContext';
import { useModelMode } from '../../store/ModelModeContext';
import { useWeatherRisk } from '../../services/weather/useWeatherRisk';

export default function GalleryScreen({ navigation }: any) {
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const { isConnected } = useNetworkStatus();
  const { t, isRTL } = useTranslation();
  const { isOnlineMode } = useModelMode();
  const { weather } = useWeatherRisk();

  const getUploadErrorMessage = (error: any) => {
    const backendMessage = error?.response?.data?.detail || error?.response?.data?.error || error?.message;
    const message = String(backendMessage || '');
    if (message.toLowerCase().includes('not a crop image') || message.toLowerCase().includes('not a leaf')) {
      return t('gallery.notCropImage');
    }
    if (message.toLowerCase().includes('timeout') || error?.code === 'ECONNABORTED') {
      return t('gallery.requestTimeout');
    }
    return message || t('gallery.uploadFailed');
  };

  // The backend handles both online (ResNet) and offline (MobileNet) modes.
  // We only queue for later if there is NO physical internet connection.

  const pickImage = async () => {
    setErrorMessage(null);
    setStatusMessage(null);
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: true,
      aspect: [4, 3],
      quality: 0.8,
    });
    if (!result.canceled) {
      setImageUri(result.assets[0].uri);
    }
  };

  const uploadImage = async () => {
    setErrorMessage(null);
    setStatusMessage(null);
    if (!imageUri) {
      const message = t('gallery.noImageSelected');
      setErrorMessage(message);
      Alert.alert(t('common.error'), message);
      return;
    }
    console.log('[Gallery] uploadImage called, isConnected:', isConnected, 'isOnlineMode:', isOnlineMode, 'platform:', Platform.OS);
    if (!isConnected) {
      const id = `offline_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      await enqueueOfflinePrediction({ id, imageUri, sourceType: 'gallery', timestamp: new Date().toISOString() });
      Alert.alert(t('gallery.savedOffline'), t('gallery.savedOfflineMsg'), [{ text: t('common.ok'), onPress: () => navigation.goBack() }]);
      return;
    }
    try {
      setIsProcessing(true);
      setStatusMessage(t('gallery.analyzingImage'));
      console.log('[Gallery] Uploading image...');
      const formData = new FormData();

      if (Platform.OS === 'web') {
        // On web, imageUri is a blob URL — fetch it and create a proper File
        const response = await fetch(imageUri);
        const blob = await response.blob();
        const filename = 'upload.jpg';
        const file = new File([blob], filename, { type: blob.type || 'image/jpeg' });
        formData.append('image', file);
      } else {
        // On native (iOS/Android), use the RN-style object
        const filename = imageUri.split('/').pop() || 'gallery.jpg';
        const match = /\.(\w+)$/.exec(filename);
        const fileType = match ? `image/${match[1]}` : 'image/jpeg';
        formData.append('image', { uri: imageUri, name: filename, type: fileType } as any);
      }

      formData.append('source_type', 'gallery');
      formData.append('mode', isOnlineMode ? 'online' : 'offline');

      // Attach weather data for enhanced online prediction
      if (weather) {
        formData.append('temperature', String(weather.temperature));
        formData.append('humidity', String(weather.humidity));
        formData.append('wind_speed', String(weather.windSpeed));
        formData.append('weather_description', weather.description);
        formData.append('weather_risk_level', weather.riskLevel);
        formData.append('city_name', weather.cityName);
      }

      const res = await apiClient.post('/predict/', formData);
      console.log('[Gallery] Prediction response:', res.data);
      setStatusMessage(null);
      navigation.navigate('Result', { prediction: res.data });
    } catch (error: any) {
      console.error('[Gallery] Upload error:', error?.response?.data || error?.message || error);
      const message = getUploadErrorMessage(error);
      setStatusMessage(null);
      setErrorMessage(message);
      Alert.alert(t('common.error'), message);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={[styles.header, isRTL && styles.rtlRow]}>
        <TouchableOpacity style={styles.iconButton} onPress={() => navigation.goBack()}>
          <ArrowLeft color="#333" size={28} style={isRTL && styles.rtlIcon} />
        </TouchableOpacity>
        <Text style={[styles.title, isRTL && styles.rtlText]}>{t('gallery.title')}</Text>
        <View style={{ width: 44 }} />
      </View>

      <View style={styles.content}>
        {!isConnected && (
          <View style={styles.offlineBanner}>
            <Text style={styles.offlineText}>{t('gallery.offlineBanner')}</Text>
          </View>
        )}
      <View style={styles.imageContainer}>
        {imageUri ? (
          <Image source={{ uri: imageUri }} style={styles.previewImage} />
        ) : (
          <View style={styles.placeholderContainer}>
            <Text style={styles.placeholderText}>{t('gallery.noImage')}</Text>
          </View>
        )}
      </View>
      <TouchableOpacity style={styles.buttonSecondary} onPress={pickImage} disabled={isProcessing}>
        <Text style={styles.buttonSecondaryText}>{imageUri ? t('gallery.changePhoto') : t('gallery.selectPhoto')}</Text>
      </TouchableOpacity>
      <TouchableOpacity style={[styles.buttonPrimary, (!imageUri || isProcessing) && styles.disabledButton]} onPress={uploadImage} disabled={isProcessing}>
        {isProcessing ? <ActivityIndicator color="white" /> : <Text style={styles.buttonPrimaryText}>{isConnected ? t('gallery.analyzeCrop') : t('gallery.saveForLater')}</Text>}
      </TouchableOpacity>
      {statusMessage && <Text style={styles.statusText}>{statusMessage}</Text>}
      {errorMessage && <Text style={styles.errorText}>{errorMessage}</Text>}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fff' },
  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 20, paddingTop: 10, paddingBottom: 20 },
  rtlRow: { flexDirection: 'row-reverse' },
  iconButton: { padding: 8 },
  rtlIcon: { transform: [{ scaleX: -1 }] },
  content: { flex: 1, paddingHorizontal: 20, alignItems: 'center' },
  title: { fontSize: 24, fontWeight: 'bold', color: '#2E7D32' },
  rtlText: { textAlign: 'right' },
  offlineBanner: { backgroundColor: '#FFF3E0', padding: 10, borderRadius: 8, width: '100%', marginBottom: 16, alignItems: 'center' },
  offlineText: { color: '#E65100', fontSize: 13, fontWeight: '600' },
  imageContainer: { width: '100%', height: 300, backgroundColor: '#F5F5F5', borderRadius: 16, overflow: 'hidden', marginBottom: 30, borderWidth: 1, borderColor: '#E0E0E0' },
  previewImage: { width: '100%', height: '100%', resizeMode: 'cover' },
  placeholderContainer: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  placeholderText: { color: '#9E9E9E', fontSize: 16 },
  buttonSecondary: { width: '100%', padding: 16, borderRadius: 8, borderWidth: 2, borderColor: '#2E7D32', alignItems: 'center', marginBottom: 16 },
  buttonSecondaryText: { color: '#2E7D32', fontSize: 16, fontWeight: 'bold' },
  buttonPrimary: { width: '100%', padding: 16, borderRadius: 8, backgroundColor: '#2E7D32', alignItems: 'center' },
  buttonPrimaryText: { color: 'white', fontSize: 16, fontWeight: 'bold' },
  disabledButton: { backgroundColor: '#A5D6A7' },
  statusText: { width: '100%', marginTop: 12, color: '#2E7D32', fontSize: 14, textAlign: 'center', fontWeight: '600' },
  errorText: { width: '100%', marginTop: 12, color: '#C62828', fontSize: 14, textAlign: 'center', fontWeight: '600' },
});
