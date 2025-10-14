import React from 'react';
import Form from '@rjsf/core';
import validator from '@rjsf/validator-ajv8';
import type { RJSFSchema, UiSchema, ValidatorType } from '@rjsf/utils';
import { PanelFrame } from '../layout/PanelFrame';
import { useControlPanels } from '../../state/AppState';
import { BeamSplitterEditor } from './BeamSplitterEditor';
import { AutotunePanel } from './AutotunePanel';
import { useI18n } from '../../i18n/LocalizationProvider';

type FormData = Record<string, unknown>;
const formValidator: ValidatorType<FormData> = validator as ValidatorType<FormData>;

export function InspectorPanel() {
  const { panels, updateForm, setCollapsed } = useControlPanels();
  const { t } = useI18n();

  if (panels.length === 0) {
    return (
      <PanelFrame title={t('inspector.title')}>
        <p className="panel-muted">{t('inspector.empty')}</p>
      </PanelFrame>
    );
  }

  return (
    <PanelFrame title={t('inspector.title')}>
      <div className="inspector inspector--forms">
        {panels.map((panel) => {
          const schema = panel.schema as RJSFSchema;
          const uiSchema = (panel.uiSchema ?? {}) as UiSchema<FormData>;
          const isCollapsed = panel.collapsed === true;
          return (
            <section key={panel.id} className="inspector-panel">
              <header className="inspector-panel__header">
                <div className="inspector-panel__title">
                  <h3>{panel.label}</h3>
                  {panel.description ? (
                    <p className="inspector-panel__description">{panel.description}</p>
                  ) : null}
                </div>
                <button
                  type="button"
                  className="ghost-button"
                  onClick={() => setCollapsed(panel.id, !isCollapsed)}
                  aria-expanded={!isCollapsed}
                >
                  {isCollapsed ? t('inspector.expand') : t('inspector.collapse')}
                </button>
              </header>
              {!isCollapsed ? (
                <Form<FormData>
                  key={`${panel.id}-form`}
                  schema={schema}
                  uiSchema={uiSchema}
                  formData={panel.formData}
                  validator={formValidator}
                  liveValidate
                  noHtml5Validate
                  idPrefix={`panel-${panel.id}`}
                  onChange={(event) => updateForm(panel.id, (event.formData ?? {}) as FormData)}
                  showErrorList={false}
                >
                  <></>
                </Form>
              ) : null}
            </section>
          );
        })}
        <BeamSplitterEditor />
        <AutotunePanel />
      </div>
    </PanelFrame>
  );
}
