import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ActivityIndicator } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { MapPin } from 'lucide-react-native';
import { useAuth } from '../../store/AuthContext';
import { useTranslation } from '../../store/LanguageContext';
import { useWeatherRisk } from '../../services/weather/useWeatherRisk';

export default function ProfileScreen({ navigation }: any) {
  const { user, logout } = useAuth();
  const { t, isRTL } = useTranslation();
  const { location, loading: locationLoading } = useWeatherRisk();

  return (
    <SafeAreaView style={styles.container}>
      <Text style={[styles.title, isRTL && styles.rtlText]}>{t('profile.title')}</Text>

      <View style={styles.infoCard}>
        <Text style={[styles.label, isRTL && styles.rtlText]}>{t('profile.name')}</Text>
        <Text style={[styles.value, isRTL && styles.rtlText]}>{user?.full_name || 'N/A'}</Text>

        <Text style={[styles.label, isRTL && styles.rtlText]}>{t('profile.email')}</Text>
        <Text style={[styles.value, isRTL && styles.rtlText]}>{user?.email || 'N/A'}</Text>

        <Text style={[styles.label, isRTL && styles.rtlText]}>{t('profile.phone')}</Text>
        <Text style={[styles.value, isRTL && styles.rtlText]}>{user?.phone_number || 'N/A'}</Text>
      </View>

      {/* Location Info */}
      <View style={styles.locationCard}>
        <View style={[styles.locationHeader, isRTL && styles.rtlRow]}>
          <MapPin color="#2E7D32" size={20} />
          <Text style={[styles.locationTitle, isRTL && { marginRight: 8, marginLeft: 0 }]}>
            {isRTL ? 'الموقع' : 'Location'}
          </Text>
        </View>
        {locationLoading ? (
          <ActivityIndicator color="#2E7D32" style={{ marginTop: 8 }} />
        ) : location ? (
          <View style={styles.locationDetails}>
            <View style={[styles.locationRow, isRTL && styles.rtlRow]}>
              <Text style={styles.locationLabel}>{isRTL ? 'المدينة' : 'City'}</Text>
              <Text style={styles.locationValue}>{location.cityName}, {location.country}</Text>
            </View>
            <View style={[styles.locationRow, isRTL && styles.rtlRow]}>
              <Text style={styles.locationLabel}>{isRTL ? 'خط العرض' : 'Latitude'}</Text>
              <Text style={styles.locationValue}>{location.latitude.toFixed(4)}</Text>
            </View>
            <View style={[styles.locationRow, isRTL && styles.rtlRow]}>
              <Text style={styles.locationLabel}>{isRTL ? 'خط الطول' : 'Longitude'}</Text>
              <Text style={styles.locationValue}>{location.longitude.toFixed(4)}</Text>
            </View>
          </View>
        ) : (
          <Text style={styles.locationUnavailable}>{isRTL ? 'الموقع غير متوفر' : 'Location unavailable'}</Text>
        )}
      </View>

      {user?.is_admin && (
        <TouchableOpacity style={styles.adminButton} onPress={() => navigation.navigate('AdminDashboard')}>
          <Text style={styles.adminButtonText}>{t('profile.adminDashboard')}</Text>
        </TouchableOpacity>
      )}

      <TouchableOpacity style={styles.logoutButton} onPress={logout}>
        <Text style={styles.logoutText}>{t('profile.logOut')}</Text>
      </TouchableOpacity>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F5F5F5', padding: 20 },
  title: { fontSize: 28, fontWeight: 'bold', color: '#2E7D32', marginBottom: 20 },
  rtlText: { textAlign: 'right' },
  rtlRow: { flexDirection: 'row-reverse' },
  infoCard: {
    backgroundColor: 'white', padding: 20, borderRadius: 12,
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.1, shadowRadius: 4, elevation: 3,
    marginBottom: 16,
  },
  label: { fontSize: 14, color: '#666', marginTop: 10 },
  value: { fontSize: 18, color: '#333', fontWeight: '500', marginBottom: 5 },
  locationCard: {
    backgroundColor: 'white', padding: 16, borderRadius: 12,
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.1, shadowRadius: 4, elevation: 3,
    marginBottom: 20,
  },
  locationHeader: { flexDirection: 'row', alignItems: 'center', marginBottom: 10 },
  locationTitle: { fontSize: 18, fontWeight: 'bold', color: '#2E7D32', marginLeft: 8 },
  locationDetails: { gap: 8 },
  locationRow: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 4 },
  locationLabel: { fontSize: 14, color: '#666' },
  locationValue: { fontSize: 14, fontWeight: '600', color: '#333' },
  locationUnavailable: { fontSize: 14, color: '#999', marginTop: 8 },
  logoutButton: { backgroundColor: '#E53935', padding: 15, borderRadius: 8, alignItems: 'center' },
  logoutText: { color: 'white', fontSize: 16, fontWeight: 'bold' },
  adminButton: { backgroundColor: '#1565C0', padding: 15, borderRadius: 8, alignItems: 'center', marginBottom: 12 },
  adminButtonText: { color: 'white', fontSize: 16, fontWeight: 'bold' },
});
