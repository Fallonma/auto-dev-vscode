import vscode from "vscode";

import { AutoDevExtension } from "../AutoDevExtension";
import { SUPPORTED_LANGUAGES } from "../editor/language/SupportedLanguage";
import { AutoDevCodeLensProvider } from "./providers/AutoDevCodeLensProvider";
import { AutoDevCodeActionProvider } from "./providers/AutoDevCodeActionProvider";
import { AutoDevCodeSuggestionProvider } from './providers/AutoDevCodeSuggestionProvider';
import { AutoDevQuickFixProvider } from "./providers/AutoDevQuickFixProvider";
import { AutoDevRenameProvider } from "./refactor/rename/AutoDevRenameProvider";
import { SettingService } from "../settings/SettingService";

export function registerCodeLensProviders(context: AutoDevExtension) {
	const filter = SUPPORTED_LANGUAGES.map(it => ({ language: it } as vscode.DocumentFilter));
	const codelensProviderSub = vscode.languages.registerCodeLensProvider(
		filter,
		new AutoDevCodeLensProvider(context),
	);

	context.extensionContext.subscriptions.push(codelensProviderSub);
}

export function registerAutoDevProviders(context: AutoDevExtension) {
	SUPPORTED_LANGUAGES.forEach((language) => {
		vscode.languages.registerCodeActionsProvider({ language },
			new AutoDevCodeActionProvider(context),
			{
				providedCodeActionKinds:
				AutoDevCodeActionProvider.providedCodeActionKinds,
			}
		);
	});
}

export function registerQuickFixProvider(context: AutoDevExtension) {
	SUPPORTED_LANGUAGES.forEach((language) => {
		vscode.languages.registerCodeActionsProvider({ language },
			new AutoDevQuickFixProvider(),
			{
				providedCodeActionKinds: AutoDevQuickFixProvider.providedCodeActionKinds,
			}
		);
	});
}

export function registerWebViewProvider(extension: AutoDevExtension) {
	extension.extensionContext.subscriptions.push(vscode.window.registerWebviewViewProvider("autodev.autodevGUIView",
			extension.sidebar, { webviewOptions: { retainContextWhenHidden: true }, }
		)
	);
}

export function registerRenameAction(extension: AutoDevExtension) {
	extension.extensionContext.subscriptions.push(vscode.languages.registerRenameProvider(SUPPORTED_LANGUAGES,
			new AutoDevRenameProvider()
		)
	);
}

export function registerCodeSuggestionProvider(extension: AutoDevExtension) {
	extension.extensionContext.subscriptions.push(vscode.languages.registerInlineCompletionItemProvider(SUPPORTED_LANGUAGES,
			new AutoDevCodeSuggestionProvider()
		)
	);
}

export function configRename(extension: AutoDevExtension) {
	if (SettingService.instance().isEnableRename()) {
		registerRenameAction(extension);
	}

	vscode.workspace.onDidChangeConfiguration(() => {
		if (SettingService.instance().isEnableRename()) {
			// 如果启用了重命名功能，则注册重命名动作（待优化）
			registerRenameAction(extension);
		} else {
			// 否则不做任何操作
		}
	});
}