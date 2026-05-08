import React from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { CheckCircle, AlertTriangle, ArrowLeft } from 'lucide-react-native';
import { useTranslation } from '../../store/LanguageContext';

export default function ResultScreen({ route, navigation }: any) {
  const { prediction } = route.params || {};
  const { t, language, isRTL } = useTranslation();

  if (!prediction) {
    return (
      <SafeAreaView style={styles.container}>
        <Text style={styles.errorText}>{t('result.noData')}</Text>
        <TouchableOpacity style={styles.backButton} onPress={() => navigation.goBack()}>
          <Text style={styles.backButtonText}>{t('common.goBack')}</Text>
        </TouchableOpacity>
      </SafeAreaView>
    );
  }

  const { is_healthy, confidence, disease_info, prediction_key } = prediction;
  const isSafe = is_healthy;
  const fallbackParts = String(prediction_key || '').split('___');
  const fallbackCropName = fallbackParts[0] || t('common.noData');
  const fallbackDiseaseName = (fallbackParts[1] || '').replace(/_/g, ' ') || t('common.noData');

  // Use language-specific fields from the disease_info
  // Default is Arabic; when English, show both English + Arabic
  const cropName = disease_info
    ? (language === 'ar'
        ? (disease_info.crop_name_ar || disease_info.crop_name_en)
        : `${disease_info.crop_name_en} - ${disease_info.crop_name_ar || ''}`.trim())
    : fallbackCropName;
  const diseaseName = disease_info
    ? (language === 'ar'
        ? (disease_info.disease_name_ar || disease_info.disease_name_en)
        : `${disease_info.disease_name_en} - ${disease_info.disease_name_ar || ''}`.trim())
    : fallbackDiseaseName;

  return (
    <SafeAreaView style={styles.container}>
      <View style={[styles.header, isRTL && styles.rtlRow]}>
        <TouchableOpacity style={styles.iconButton} onPress={() => navigation.goBack()}>
          <ArrowLeft color="#333" size={24} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>{t('result.title')}</Text>
        <View style={{ width: 40 }} />
      </View>

      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={[styles.statusCard, isSafe ? styles.statusSafe : styles.statusDanger]}>
          {isSafe ? <CheckCircle color="#2E7D32" size={48} /> : <AlertTriangle color="#D32F2F" size={48} />}
          <Text style={[styles.statusTitle, { color: isSafe ? '#2E7D32' : '#D32F2F' }]}>
            {isSafe ? t('result.healthyCrop') : t('result.diseaseDetected')}
          </Text>

        </View>

        <View style={styles.infoSection}>
          <Text style={[styles.sectionLabel, isRTL && styles.rtlText]}>{t('result.cropIdentified')}</Text>
          <Text style={[styles.sectionValue, isRTL && styles.rtlText]}>{cropName}</Text>

          {!isSafe && (
            <>
              <View style={styles.divider} />
              <Text style={[styles.sectionLabel, isRTL && styles.rtlText]}>{t('result.diseaseType')}</Text>
              <Text style={[styles.sectionValue, isRTL && styles.rtlText]}>{diseaseName}</Text>
            </>
          )}
        </View>
      </ScrollView>

      <View style={styles.footer}>
        <TouchableOpacity style={styles.doneButton} onPress={() => navigation.goBack()}>
          <Text style={styles.doneText}>{t('common.done')}</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#FAFAFA' },
  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', padding: 16, backgroundColor: '#fff', borderBottomWidth: 1, borderBottomColor: '#EEE' },
  rtlRow: { flexDirection: 'row-reverse' },
  rtlText: { textAlign: 'right' },
  headerTitle: { fontSize: 18, fontWeight: 'bold', color: '#333' },
  iconButton: { padding: 8 },
  scrollContent: { padding: 16 },
  statusCard: { padding: 24, borderRadius: 16, alignItems: 'center', marginBottom: 20, borderWidth: 2 },
  statusSafe: { backgroundColor: '#E8F5E9', borderColor: '#C8E6C9' },
  statusDanger: { backgroundColor: '#FFEBEE', borderColor: '#FFCDD2' },
  statusTitle: { fontSize: 24, fontWeight: 'bold', marginTop: 12 },
  statusSubtitle: { fontSize: 16, color: '#666', marginTop: 4 },
  infoSection: { backgroundColor: '#fff', borderRadius: 16, padding: 20, shadowColor: '#000', shadowOpacity: 0.05, shadowRadius: 10, elevation: 2 },
  sectionLabel: { fontSize: 12, color: '#888', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 },
  sectionValue: { fontSize: 18, fontWeight: 'bold', color: '#333', marginBottom: 6 },
  bodyText: { fontSize: 15, color: '#444', lineHeight: 22 },
  divider: { height: 1, backgroundColor: '#EEE', marginVertical: 16 },
  footer: { padding: 20, backgroundColor: '#fff', borderTopWidth: 1, borderTopColor: '#EEE' },
  doneButton: { backgroundColor: '#2E7D32', padding: 16, borderRadius: 12, alignItems: 'center' },
  doneText: { color: 'white', fontSize: 18, fontWeight: 'bold' },
  errorText: { fontSize: 18, color: '#D32F2F', textAlign: 'center', marginTop: 40 },
  backButton: { margin: 20, padding: 15, backgroundColor: '#E0E0E0', borderRadius: 8, alignItems: 'center' },
  backButtonText: { fontSize: 16, fontWeight: 'bold', color: '#333' },
});
