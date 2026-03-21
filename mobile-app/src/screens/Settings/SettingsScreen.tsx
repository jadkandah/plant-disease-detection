import React, { useState } from 'react';
import { View, Text, StyleSheet, Switch, TouchableOpacity, ScrollView, Alert } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { ChevronRight, HelpCircle, Info } from 'lucide-react-native';
import { useTranslation } from '../../store/LanguageContext';

export default function SettingsScreen() {
  const [notificationsEnabled, setNotificationsEnabled] = useState(true);
  const { t, language, setLanguage, isRTL } = useTranslation();

  const handleHelpSupport = () => {
    Alert.alert(
      t('settings.helpSupport'),
      isRTL ? 'تواصل معنا عبر البريد الإلكتروني: support@plantdisease.com' : 'Contact us via email at: support@plantdisease.com'
    );
  };

  const handleAbout = () => {
    Alert.alert(
      t('settings.about'),
      `${t('settings.version')}\n${t('settings.gradProject')}`
    );
  };

  return (
    <SafeAreaView style={styles.container}>
      <Text style={[styles.title, isRTL && styles.rtlText]}>{t('settings.title')}</Text>
      <ScrollView>
        <View style={[styles.settingRow, isRTL && styles.rtlRow]}>
          <Text style={styles.settingLabel}>{t('settings.pushNotifications')}</Text>
          <Switch
            trackColor={{ false: '#ccc', true: '#81C784' }}
            thumbColor={notificationsEnabled ? '#2E7D32' : '#f4f3f4'}
            onValueChange={setNotificationsEnabled}
            value={notificationsEnabled}
          />
        </View>

        <View style={[styles.settingRow, isRTL && styles.rtlRow]}>
          <Text style={styles.settingLabel}>{t('settings.language')}</Text>
          <View style={[styles.languageToggle, isRTL && styles.rtlRow]}>
            <TouchableOpacity
              style={[styles.langBtn, language === 'en' && styles.langBtnActive]}
              onPress={() => setLanguage('en')}
            >
              <Text style={[styles.langText, language === 'en' && styles.langTextActive]}>EN</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.langBtn, language === 'ar' && styles.langBtnActive]}
              onPress={() => setLanguage('ar')}
            >
              <Text style={[styles.langText, language === 'ar' && styles.langTextActive]}>AR</Text>
            </TouchableOpacity>
          </View>
        </View>

        <View style={styles.actionGroup}>
          <TouchableOpacity style={[styles.actionRow, isRTL && styles.rtlRow]} onPress={handleHelpSupport}>
            <View style={[styles.actionLeft, isRTL && styles.rtlRow]}>
              <HelpCircle color="#2E7D32" size={24} />
              <Text style={[styles.actionLabel, isRTL ? { marginRight: 12 } : { marginLeft: 12 }]}>{t('settings.helpSupport')}</Text>
            </View>
            <ChevronRight color="#ccc" size={20} style={isRTL && styles.rtlIcon} />
          </TouchableOpacity>
          
          <View style={styles.divider} />
          
          <TouchableOpacity style={[styles.actionRow, isRTL && styles.rtlRow]} onPress={handleAbout}>
            <View style={[styles.actionLeft, isRTL && styles.rtlRow]}>
              <Info color="#2E7D32" size={24} />
              <Text style={[styles.actionLabel, isRTL ? { marginRight: 12 } : { marginLeft: 12 }]}>{t('settings.about')}</Text>
            </View>
            <ChevronRight color="#ccc" size={20} style={isRTL && styles.rtlIcon} />
          </TouchableOpacity>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F5F5F5', padding: 20 },
  title: { fontSize: 28, fontWeight: 'bold', color: '#2E7D32', marginBottom: 20 },
  rtlText: { textAlign: 'right' },
  rtlRow: { flexDirection: 'row-reverse' },
  rtlIcon: { transform: [{ scaleX: -1 }] },
  settingRow: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
    backgroundColor: 'white', padding: 16, borderRadius: 12, marginBottom: 12,
    shadowColor: '#000', shadowOpacity: 0.05, shadowRadius: 4, elevation: 2,
  },
  settingLabel: { fontSize: 16, color: '#333' },
  languageToggle: { flexDirection: 'row', gap: 8 },
  langBtn: { paddingVertical: 6, paddingHorizontal: 16, borderRadius: 6, borderWidth: 1, borderColor: '#ccc' },
  langBtnActive: { backgroundColor: '#2E7D32', borderColor: '#2E7D32' },
  langText: { fontSize: 14, color: '#666' },
  langTextActive: { color: 'white', fontWeight: 'bold' },
  actionGroup: {
    backgroundColor: 'white', borderRadius: 12, marginTop: 20, shadowColor: '#000', shadowOpacity: 0.05, shadowRadius: 4, elevation: 2,
    overflow: 'hidden'
  },
  actionRow: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', padding: 16
  },
  actionLeft: {
    flexDirection: 'row', alignItems: 'center'
  },
  actionLabel: {
    fontSize: 16, color: '#333'
  },
  divider: {
    height: 1, backgroundColor: '#F0F0F0', marginHorizontal: 16
  }
});
