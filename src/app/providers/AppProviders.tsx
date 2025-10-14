import React from 'react';
import { AppStateProvider } from '../../state/AppState';
import { LocalizationProvider } from '../../i18n/LocalizationProvider';
import { AccessibilityPreferencesProvider } from './AccessibilityPreferences';

export interface AppProvidersProps {
  readonly children: React.ReactNode;
}

export function AppProviders({ children }: AppProvidersProps) {
  return (
    <LocalizationProvider>
      <AccessibilityPreferencesProvider>
        <AppStateProvider>{children}</AppStateProvider>
      </AccessibilityPreferencesProvider>
    </LocalizationProvider>
  );
}
