import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ScrollView } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { HeartPulse, Camera, Image as ImageIcon, Wifi, WifiOff, Cloud, CloudOff } from 'lucide-react-native';
import { useTranslation } from '../../store/LanguageContext';
import { useNetworkStatus } from '../../services/network/useNetworkStatus';
import { useModelMode } from '../../store/ModelModeContext';

const DAILY_TIP_KEYS = [
  'home.tipText1',
  'home.tipText2',
  'home.tipText3',
  'home.tipText4',
  'home.tipText5',
  'home.tipText6',
  'home.tipText7',
];

const getDailyTipKey = () => {
  const today = new Date();
  const dayNumber = Math.floor(
    Date.UTC(today.getFullYear(), today.getMonth(), today.getDate()) / 86400000
  );

  return DAILY_TIP_KEYS[dayNumber % DAILY_TIP_KEYS.length];
};

export default function HomeScreen({ navigation }: any) {
  const { t, isRTL } = useTranslation();
  const { isConnected } = useNetworkStatus();
  const { isOnlineMode } = useModelMode();
  const dailyTipKey = getDailyTipKey();

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>

        <View style={[styles.connectionBar, isConnected ? styles.connectedBar : styles.disconnectedBar, isRTL && styles.rtlRow]}>
          {isConnected ? <Wifi color="#2E7D32" size={16} /> : <WifiOff color="#C62828" size={16} />}
          <Text style={[styles.connectionText, { color: isConnected ? '#2E7D32' : '#C62828' }]}>
            {isConnected ? (isRTL ? 'متصل بالإنترنت' : 'Connected to Internet') : (isRTL ? 'غير متصل بالإنترنت' : 'No Internet Connection')}
          </Text>
        </View>


        <View style={[styles.connectionBar, isOnlineMode ? styles.onlineModeBar : styles.offlineModeBar, isRTL && styles.rtlRow]}>
          {isOnlineMode ? <Cloud color="#1565C0" size={16} /> : <CloudOff color="#E65100" size={16} />}
          <Text style={[styles.connectionText, { color: isOnlineMode ? '#1565C0' : '#E65100' }]}>
            {isOnlineMode ? t('home.modelOnline') : t('home.modelOffline')}
          </Text>
        </View>

        <View style={styles.header}>
          <Text style={[styles.greeting, isRTL && styles.rtlText]}>{t('home.greeting')}</Text>
          <Text style={[styles.title, isRTL && styles.rtlText]}>{t('home.title')}</Text>
        </View>


        <View style={[styles.actionContainer, isRTL && styles.rtlRow]}>
          <TouchableOpacity style={[styles.actionCard, { backgroundColor: '#E8F5E9' }]} onPress={() => navigation.navigate('Camera')}>
            <Camera color="#2E7D32" size={32} />
            <Text style={styles.cardTitle}>{t('home.takePhoto')}</Text>
            <Text style={styles.cardSubtitle}>{t('home.cameraSubtitle')}</Text>
          </TouchableOpacity>

          <TouchableOpacity style={[styles.actionCard, { backgroundColor: '#F3E5F5' }]} onPress={() => navigation.navigate('Gallery')}>
            <ImageIcon color="#6A1B9A" size={32} />
            <Text style={styles.cardTitle}>{t('home.uploadGallery')}</Text>
            <Text style={styles.cardSubtitle}>{t('home.gallerySubtitle')}</Text>
          </TouchableOpacity>
        </View>


        <View style={[styles.tipContainer, isRTL && styles.rtlRow]}>
          <HeartPulse color="#E65100" size={24} style={isRTL ? { marginLeft: 16 } : { marginRight: 16 }} />
          <View style={styles.tipTextContainer}>
            <Text style={[styles.tipTitle, isRTL && styles.rtlText]}>{t('home.dailyTip')}</Text>
            <Text style={[styles.tipText, isRTL && styles.rtlText]}>{t(dailyTipKey)}</Text>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fff' },
  scrollContent: { padding: 20 },
  connectionBar: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 10,
    marginBottom: 16,
    gap: 8,
  },
  connectedBar: { backgroundColor: '#E8F5E9' },
  disconnectedBar: { backgroundColor: '#FFEBEE' },
  onlineModeBar: { backgroundColor: '#E3F2FD' },
  offlineModeBar: { backgroundColor: '#FFF3E0' },
  connectionText: { fontSize: 13, fontWeight: '600' },
  header: { marginBottom: 20 },
  greeting: { fontSize: 16, color: '#666' },
  title: { fontSize: 28, fontWeight: 'bold', color: '#2E7D32', marginTop: 5 },
  rtlText: { textAlign: 'right' },
  rtlRow: { flexDirection: 'row-reverse' },
  actionContainer: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 24 },
  actionCard: {
    width: '48%', padding: 20, borderRadius: 16, alignItems: 'center', justifyContent: 'center',
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.1, shadowRadius: 4, elevation: 3,
  },
  cardTitle: { fontSize: 16, fontWeight: 'bold', color: '#333', marginTop: 12, textAlign: 'center' },
  cardSubtitle: { fontSize: 12, color: '#666', marginTop: 4, textAlign: 'center' },
  tipContainer: { backgroundColor: '#FFF3E0', borderRadius: 12, padding: 16, flexDirection: 'row', alignItems: 'center' },
  tipTextContainer: { flex: 1 },
  tipTitle: { fontSize: 16, fontWeight: 'bold', color: '#E65100', marginBottom: 4 },
  tipText: { fontSize: 14, color: '#333', lineHeight: 20 },
});
