import React, { useState, useCallback } from 'react';
import { View, Text, StyleSheet, ScrollView, ActivityIndicator } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useFocusEffect } from '@react-navigation/native';
import apiClient from '../../services/auth/apiClient';
import { useTranslation } from '../../store/LanguageContext';

interface Stats {
  total_users: number;
  total_predictions: number;
  diseased_detections: number;
  healthy_detections: number;
}

interface PredictionItem {
  id: number;
  crop_name: string;
  disease_name_en: string;
  confidence: number;
  is_healthy: boolean;
  predicted_at: string;
}

export default function AdminDashboardScreen() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [recentPredictions, setRecentPredictions] = useState<PredictionItem[]>([]);
  const [loading, setLoading] = useState(true);
  const { t, isRTL } = useTranslation();

  useFocusEffect(useCallback(() => { fetchAdminData(); }, []));

  const fetchAdminData = async () => {
    try {
      setLoading(true);
      const [statsRes, predsRes] = await Promise.all([
        apiClient.get('/admin/predictions/stats/'),
        apiClient.get('/admin/predictions/?limit=5'),
      ]);
      setStats(statsRes.data);
      setRecentPredictions(predsRes.data.results || predsRes.data || []);
    } catch (error) {
      setStats({ total_users: 0, total_predictions: 0, diseased_detections: 0, healthy_detections: 0 });
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <SafeAreaView style={styles.container}>
        <ActivityIndicator size="large" color="#2E7D32" style={{ flex: 1 }} />
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <Text style={[styles.title, isRTL && styles.rtlText]}>{t('admin.title')}</Text>
      <ScrollView>
        <View style={[styles.statsGrid, isRTL && styles.rtlRow]}>
          <View style={[styles.statCard, { backgroundColor: '#E3F2FD' }]}>
            <Text style={styles.statNumber}>{stats?.total_users ?? 0}</Text>
            <Text style={styles.statLabel}>{t('admin.totalUsers')}</Text>
          </View>
          <View style={[styles.statCard, { backgroundColor: '#E8F5E9' }]}>
            <Text style={styles.statNumber}>{stats?.total_predictions ?? 0}</Text>
            <Text style={styles.statLabel}>{t('admin.totalScans')}</Text>
          </View>
          <View style={[styles.statCard, { backgroundColor: '#FFF3E0' }]}>
            <Text style={styles.statNumber}>{stats?.diseased_detections ?? 0}</Text>
            <Text style={styles.statLabel}>{t('admin.diseasesFound')}</Text>
          </View>
          <View style={[styles.statCard, { backgroundColor: '#FCE4EC' }]}>
            <Text style={styles.statNumber}>{stats?.healthy_detections ?? 0}</Text>
            <Text style={styles.statLabel}>{t('admin.healthyCrops')}</Text>
          </View>
        </View>

        <Text style={[styles.sectionTitle, isRTL && styles.rtlText]}>{t('admin.recentPredictions')}</Text>
        {recentPredictions.length === 0 ? (
          <Text style={[styles.emptyText, isRTL && styles.rtlText]}>{t('admin.noPredictions')}</Text>
        ) : (
          recentPredictions.slice(0, 5).map((item) => (
            <View key={item.id} style={[styles.predictionRow, isRTL && styles.rtlRow]}>
              <View style={[styles.dot, { backgroundColor: item.is_healthy ? '#2E7D32' : '#D32F2F' }]} />
              <View style={{ flex: 1 }}>
                <Text style={[styles.predCrop, isRTL && styles.rtlText]}>{item.crop_name} — {item.disease_name_en}</Text>
                <Text style={[styles.predDate, isRTL && styles.rtlText]}>
                  {new Date(item.predicted_at).toLocaleDateString()} · {(item.confidence * 100).toFixed(0)}%
                </Text>
              </View>
            </View>
          ))
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F5F5F5', padding: 20 },
  title: { fontSize: 28, fontWeight: 'bold', color: '#2E7D32', marginBottom: 20 },
  rtlText: { textAlign: 'right' },
  rtlRow: { flexDirection: 'row-reverse' },
  statsGrid: { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'space-between', marginBottom: 24 },
  statCard: { width: '48%', padding: 20, borderRadius: 16, alignItems: 'center', marginBottom: 12, shadowColor: '#000', shadowOpacity: 0.05, shadowRadius: 4, elevation: 2 },
  statNumber: { fontSize: 32, fontWeight: 'bold', color: '#333' },
  statLabel: { fontSize: 13, color: '#666', marginTop: 6, textAlign: 'center' },
  sectionTitle: { fontSize: 20, fontWeight: 'bold', color: '#333', marginBottom: 12 },
  emptyText: { fontSize: 14, color: '#999', textAlign: 'center', marginVertical: 20 },
  predictionRow: { flexDirection: 'row', alignItems: 'center', backgroundColor: 'white', padding: 14, borderRadius: 10, marginBottom: 8, shadowColor: '#000', shadowOpacity: 0.03, shadowRadius: 3, elevation: 1 },
  dot: { width: 10, height: 10, borderRadius: 5, marginRight: 12 },
  predCrop: { fontSize: 15, fontWeight: '600', color: '#333' },
  predDate: { fontSize: 12, color: '#888', marginTop: 2 },
});
