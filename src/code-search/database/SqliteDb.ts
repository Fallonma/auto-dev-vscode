import * as fs from 'fs';
import * as path from "path";
import * as sqlite3 from "sqlite3";
import * as sqlite from "sqlite";

import { getIndexFolderPath } from "../indexing/LanceDbIndex";

export type DatabaseConnection = sqlite.Database<sqlite3.Database>;

export function getIndexSqlitePath(): string {
	return path.join(getIndexFolderPath(), "autodev-index.sqlite");
}

export class SqliteDb {
	static db: DatabaseConnection | null = null;

	private static async createTables(db: DatabaseConnection) {
		await db.exec(
			`CREATE TABLE IF NOT EXISTS tag_catalog (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dir STRING NOT NULL,
            branch STRING NOT NULL,
            artifactId STRING NOT NULL,
            path STRING NOT NULL,
            cacheKey STRING NOT NULL,
            lastUpdated INTEGER NOT NULL
        )`,
		);

		await db.exec(
			`CREATE TABLE IF NOT EXISTS global_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cacheKey STRING NOT NULL,
            dir STRING NOT NULL,
            branch STRING NOT NULL,
            artifactId STRING NOT NULL
        )`,
		);
	}

	private static indexSqlitePath = getIndexSqlitePath();

	static async get() {
		if (SqliteDb.db && fs.existsSync(SqliteDb.indexSqlitePath)) {
			return SqliteDb.db;
		}

		SqliteDb.indexSqlitePath = getIndexSqlitePath();
		SqliteDb.db = await sqlite.open({
			filename: SqliteDb.indexSqlitePath,
			driver: sqlite3.Database,
		});

		await SqliteDb.createTables(SqliteDb.db);

		return SqliteDb.db;
	}
}
