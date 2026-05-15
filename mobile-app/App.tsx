import React, { useContext } from 'react';
import { View, Text, ActivityIndicator, StyleSheet } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Home, History, User, Settings, Clock3, RefreshCw, CheckCircle2, AlertTriangle } from 'lucide-react-native';
import { AutoSyncState, useAutoSync } from './src/services/sync/useAutoSync';
import { useTranslation } from './src/store/LanguageContext';

import { AuthProvider, AuthContext } from './src/store/AuthContext';
import { LanguageProvider } from './src/store/LanguageContext';
import { ModelModeProvider } from './src/store/ModelModeContext';
import { LoginScreen } from './src/screens/Auth/LoginScreen';
import { SignUpScreen } from './src/screens/Auth/SignUpScreen';


import HomeScreen from './src/screens/Home/HomeScreen';
import HistoryScreen from './src/screens/History/HistoryScreen';
import ProfileScreen from './src/screens/Profile/ProfileScreen';
import SettingsScreen from './src/screens/Settings/SettingsScreen';


import CameraScreen from './src/screens/Home/CameraScreen';
import GalleryScreen from './src/screens/Home/GalleryScreen';
import ResultScreen from './src/screens/Home/ResultScreen';
import AdminDashboardScreen from './src/screens/Admin/AdminDashboardScreen';

const Stack = createNativeStackNavigator();
const Tab = createBottomTabNavigator();

const formatCountMessage = (message: string, count: number) => message.replace('{count}', String(count));

const SyncStatusBanner = ({ state }: { state: AutoSyncState }) => {
  const { t, isRTL } = useTranslation();
  if (!state.visible || state.phase === 'idle') return null;

  const count = state.phase === 'synced' ? state.syncedCount : state.pendingCount;
  if (count <= 0) return null;

  let color = '#F57C00';
  let title = t('sync.waitingTitle');
  let message = formatCountMessage(t('sync.waitingMessage'), count);
  let icon = <Clock3 color={color} size={20} />;

  if (state.phase === 'syncing') {
    color = '#1565C0';
    title = t('sync.syncingTitle');
    message = formatCountMessage(t('sync.syncingMessage'), count);
    icon = <RefreshCw color={color} size={20} />;
  } else if (state.phase === 'synced') {
    color = '#2E7D32';
    title = t('sync.syncComplete');
    message = formatCountMessage(t('sync.syncMessage'), count);
    icon = <CheckCircle2 color={color} size={20} />;
  } else if (state.phase === 'failed') {
    color = '#C62828';
    title = t('sync.syncFailedTitle');
    message = formatCountMessage(t('sync.syncFailedMessage'), count);
    icon = <AlertTriangle color={color} size={20} />;
  }

  return (
    <View
      pointerEvents="none"
      style={[
        styles.syncBanner,
        isRTL && styles.syncBannerRtl,
        {
          borderLeftColor: isRTL ? 'transparent' : color,
          borderRightColor: isRTL ? color : 'transparent',
        },
      ]}
    >
      <View style={[styles.syncIconWrap, isRTL && styles.syncIconWrapRtl, { backgroundColor: `${color}18` }]}>
        {icon}
      </View>
      <View style={styles.syncTextBlock}>
        <Text style={[styles.syncTitle, isRTL && styles.rtlText]}>{title}</Text>
        <Text style={[styles.syncMessage, isRTL && styles.rtlText]}>{message}</Text>
      </View>
    </View>
  );
};

const MainTabs = () => {
  const { t } = useTranslation();
  return (
    <Tab.Navigator
      screenOptions={{
        tabBarActiveTintColor: '#2E7D32',
        tabBarInactiveTintColor: 'gray',
        headerShown: false,
        tabBarStyle: {
          backgroundColor: '#fff',
          borderTopWidth: 1,
          borderTopColor: '#E0E0E0',
          paddingBottom: 5,
          paddingTop: 5,
          height: 60,
        },
      }}
    >
      <Tab.Screen
        name="HomeDashboard"
        component={HomeScreen}
        options={{ tabBarLabel: t('home.takePhoto').includes('Photo') ? 'Home' : 'الرئيسية', tabBarIcon: ({ color, size }) => <Home color={color} size={size} /> }}
      />
      <Tab.Screen
        name="HistoryTab"
        component={HistoryScreen}
        options={{ tabBarLabel: t('history.title').includes('History') ? 'History' : 'السجل', tabBarIcon: ({ color, size }) => <History color={color} size={size} /> }}
      />
      <Tab.Screen
        name="ProfileTab"
        component={ProfileScreen}
        options={{ tabBarLabel: t('profile.title').includes('Profile') ? 'Profile' : 'الملف', tabBarIcon: ({ color, size }) => <User color={color} size={size} /> }}
      />
      <Tab.Screen
        name="SettingsTab"
        component={SettingsScreen}
        options={{ tabBarLabel: t('settings.title').includes('Settings') ? 'Settings' : 'الإعدادات', tabBarIcon: ({ color, size }) => <Settings color={color} size={size} /> }}
      />
    </Tab.Navigator>
  );
};

const Navigation = () => {
  const { user, isLoading } = useContext(AuthContext);
  const syncState = useAutoSync(Boolean(user));

  if (isLoading) {
    return (
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#fff' }}>
        <ActivityIndicator size="large" color="#2E7D32" />
      </View>
    );
  }

  return (
    <View style={styles.appRoot}>
      <NavigationContainer>
        <Stack.Navigator screenOptions={{ headerShown: false }}>
          {user ? (
            <>
              <Stack.Screen name="MainTabs" component={MainTabs} />
              <Stack.Screen name="Camera" component={CameraScreen} />
              <Stack.Screen name="Gallery" component={GalleryScreen} />
              <Stack.Screen name="Result" component={ResultScreen} options={{ gestureEnabled: false }} />
              <Stack.Screen name="AdminDashboard" component={AdminDashboardScreen} />
            </>
          ) : (
            <>
              <Stack.Screen name="Login" component={LoginScreen} />
              <Stack.Screen name="SignUp" component={SignUpScreen} />
            </>
          )}
        </Stack.Navigator>
      </NavigationContainer>
      {user && <SyncStatusBanner state={syncState} />}
    </View>
  );
};

export default function App() {
  return (
    <LanguageProvider>
      <ModelModeProvider>
        <AuthProvider>
          <Navigation />
        </AuthProvider>
      </ModelModeProvider>
    </LanguageProvider>
  );
}

const styles = StyleSheet.create({
  appRoot: {
    flex: 1,
  },
  syncBanner: {
    position: 'absolute',
    left: 16,
    right: 16,
    bottom: 76,
    zIndex: 1000,
    elevation: 8,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#FFFFFF',
    borderRadius: 8,
    borderLeftWidth: 4,
    borderRightWidth: 0,
    paddingVertical: 10,
    paddingHorizontal: 12,
    shadowColor: '#000000',
    shadowOpacity: 0.14,
    shadowRadius: 8,
    shadowOffset: { width: 0, height: 2 },
  },
  syncBannerRtl: {
    flexDirection: 'row-reverse',
    borderLeftWidth: 0,
    borderRightWidth: 4,
  },
  syncIconWrap: {
    width: 34,
    height: 34,
    borderRadius: 17,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 10,
  },
  syncIconWrapRtl: {
    marginRight: 0,
    marginLeft: 10,
  },
  syncTextBlock: {
    flex: 1,
  },
  syncTitle: {
    color: '#263238',
    fontSize: 14,
    fontWeight: '700',
  },
  syncMessage: {
    color: '#546E7A',
    fontSize: 12,
    marginTop: 2,
  },
  rtlText: {
    textAlign: 'right',
  },
});
