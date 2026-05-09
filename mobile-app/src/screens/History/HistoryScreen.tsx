import React, { useState, useCallback } from 'react';
import { View, Text, StyleSheet, FlatList, TextInput, TouchableOpacity, ActivityIndicator } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useFocusEffect } from '@react-navigation/native';
import apiClient from '../../services/auth/apiClient';
import { useTranslation } from '../../store/LanguageContext';

interface PredictionItem {
  id: number;
  crop_name: string;
  disease_name_en: string;
  confidence: number;
  is_healthy: boolean;
  predicted_at: string;
  source_type: 'camera' | 'gallery';
  model_mode: 'online' | 'offline';
  sync_status: 'synced' | 'pending';
}

export default function HistoryScreen() {
  const [predictions, setPredictions] = useState<PredictionItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const { t, isRTL } = useTranslation();

  const fetchHistory = async (crop?: string) => {
    try {
      setLoading(true);
      const params: any = {};
      if (crop) params.crop = crop;
      const response = await apiClient.get('/history/', { params });
      setPredictions(response.data.results || response.data || []);
    } catch (error) {
      console.log('Failed to fetch history:', error);
    } finally {
      setLoading(false);
    }
  };

  useFocusEffect(useCallback(() => { fetchHistory(); }, []));

  const handleSearch = () => { fetchHistory(searchQuery || undefined); };

  const renderItem = ({ item }: { item: PredictionItem }) => {
    const isOffline = item.model_mode === 'offline';
    const modelLabel = isOffline ? t('history.offlineSource') : t('history.onlineSource');
    const captureLabel = item.source_type === 'gallery' ? t('history.gallerySource') : t('history.cameraSource');

    return (
      <View style={[styles.card, item.is_healthy ? styles.cardHealthy : styles.cardDiseased]}>
        <View style={[styles.cardHeader, isRTL && styles.rtlRow]}>
          <Text style={styles.cropName}>{item.crop_name}</Text>
          <View style={[styles.badgeRow, isRTL && styles.rtlRow]}>
            <Text style={[styles.sourceBadge, isOffline ? styles.sourceBadgeOffline : styles.sourceBadgeOnline]}>
              {modelLabel}
            </Text>
            <Text style={[styles.badge, item.is_healthy ? styles.badgeHealthy : styles.badgeDiseased]}>
              {item.is_healthy ? t('history.healthy') : t('history.diseased')}
            </Text>
          </View>
        </View>
        {!item.is_healthy && <Text style={[styles.diseaseName, isRTL && styles.rtlText]}>{item.disease_name_en}</Text>}
        <View style={[styles.cardFooter, isRTL && styles.rtlRow]}>
          <Text style={styles.confidence}>{t('result.confidence')}: {(item.confidence * 100).toFixed(1)}%</Text>
          <Text style={styles.sourceText}>{captureLabel}</Text>
          <Text style={styles.date}>{new Date(item.predicted_at).toLocaleDateString()}</Text>
        </View>
      </View>
    );
  };

  return (
    <SafeAreaView style={styles.container}>
      <Text style={[styles.title, isRTL && styles.rtlText]}>{t('history.title')}</Text>
      <View style={[styles.searchRow, isRTL && styles.rtlRow]}>
        <TextInput style={[styles.searchInput, isRTL && { textAlign: 'right' }]} placeholder={t('history.searchPlaceholder')} value={searchQuery} onChangeText={setSearchQuery} onSubmitEditing={handleSearch} />
        <TouchableOpacity style={styles.searchButton} onPress={handleSearch}>
          <Text style={styles.searchButtonText}>{t('history.filter')}</Text>
        </TouchableOpacity>
      </View>
      {loading ? (
        <View style={styles.centerBox}><ActivityIndicator size="large" color="#2E7D32" /></View>
      ) : predictions.length === 0 ? (
        <View style={styles.centerBox}>
          <Text style={styles.emptyText}>{t('history.noPredictions')}</Text>
          <Text style={styles.emptySubtext}>{t('history.firstDiagnosis')}</Text>
        </View>
      ) : (
        <FlatList data={predictions} keyExtractor={(item) => String(item.id)} renderItem={renderItem} contentContainerStyle={{ paddingBottom: 20 }} />
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F5F5F5', padding: 20 },
  title: { fontSize: 28, fontWeight: 'bold', color: '#2E7D32', marginBottom: 16 },
  rtlText: { textAlign: 'right' },
  rtlRow: { flexDirection: 'row-reverse' },
  searchRow: { flexDirection: 'row', marginBottom: 16 },
  searchInput: { flex: 1, backgroundColor: '#fff', borderWidth: 1, borderColor: '#ccc', borderRadius: 8, paddingHorizontal: 12, paddingVertical: 10, fontSize: 15, marginRight: 8 },
  searchButton: { backgroundColor: '#2E7D32', paddingHorizontal: 16, borderRadius: 8, justifyContent: 'center' },
  searchButtonText: { color: 'white', fontWeight: 'bold', fontSize: 14 },
  centerBox: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  emptyText: { fontSize: 18, color: '#666', fontWeight: 'bold' },
  emptySubtext: { fontSize: 14, color: '#999', marginTop: 6 },
  card: { backgroundColor: '#fff', borderRadius: 12, padding: 16, marginBottom: 12, borderLeftWidth: 4, shadowColor: '#000', shadowOpacity: 0.05, shadowRadius: 4, elevation: 2 },
  cardHealthy: { borderLeftColor: '#2E7D32' },
  cardDiseased: { borderLeftColor: '#D32F2F' },
  cardHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 },
  cropName: { fontSize: 18, fontWeight: 'bold', color: '#333' },
  badgeRow: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  badge: { paddingHorizontal: 10, paddingVertical: 3, borderRadius: 12, fontSize: 12, fontWeight: 'bold', overflow: 'hidden' },
  badgeHealthy: { backgroundColor: '#E8F5E9', color: '#2E7D32' },
  badgeDiseased: { backgroundColor: '#FFEBEE', color: '#D32F2F' },
  sourceBadge: { paddingHorizontal: 10, paddingVertical: 3, borderRadius: 12, fontSize: 12, fontWeight: 'bold', overflow: 'hidden' },
  sourceBadgeOnline: { backgroundColor: '#E3F2FD', color: '#1565C0' },
  sourceBadgeOffline: { backgroundColor: '#FFF3E0', color: '#E65100' },
  diseaseName: { fontSize: 15, color: '#666', marginBottom: 8 },
  cardFooter: { flexDirection: 'row', justifyContent: 'space-between', gap: 8 },
  confidence: { fontSize: 13, color: '#888' },
  sourceText: { fontSize: 13, color: '#888' },
  date: { fontSize: 13, color: '#888' },
});
