import { exec } from "child_process";
import path from "path";
import { promisify } from "util";
import type { PythonReport } from "@/types";

const execAsync = promisify(exec);

// Resolve paths from environment variables with sensible defaults
const NEPSE_ROOT =
  process.env.NEPSE_ROOT ??
  path.resolve(process.cwd(), ".."); // frontend/../ = project root

const PYTHON_EXEC =
  process.env.PYTHON_EXEC ??
  path.join(NEPSE_ROOT, ".venv", "Scripts", "python.exe");

const PREDICT_SCRIPT = path.join(NEPSE_ROOT, "predict_api.py");

// Allow up to 11 minutes — the ML pipeline can take time
const EXEC_TIMEOUT_MS = 660_000;

// ── runPrediction ─────────────────────────────────────────────────────────────

export async function runPrediction(symbol: string): Promise<PythonReport> {
  const normalizedSymbol = symbol.toUpperCase().trim();

  // Shell-escape the symbol; NEPSE symbols are always uppercase alphanumeric
  if (!/^[A-Z0-9]{1,12}$/.test(normalizedSymbol)) {
    throw new Error(`Invalid symbol format: ${symbol}`);
  }

  const cmd = `"${PYTHON_EXEC}" "${PREDICT_SCRIPT}" --symbol ${normalizedSymbol}`;

  console.info(`[python-runner] Starting prediction for ${normalizedSymbol}`);
  const startedAt = Date.now();

  let stdout: string;
  let stderr: string;

  try {
    ({ stdout, stderr } = await execAsync(cmd, {
      timeout: EXEC_TIMEOUT_MS,
      maxBuffer: 20 * 1024 * 1024, // 20 MB
      cwd: NEPSE_ROOT,
    }));
  } catch (err: unknown) {
    // execAsync throws on non-zero exit; predict_api.py writes
    // {"error":"...","symbol":"..."} to stdout before calling sys.exit(1).
    // Extract that for a meaningful user-facing message.
    const execErr = err as { stdout?: string; stderr?: string; message?: string };
    const rawOut = (execErr.stdout ?? "").trim();

    if (rawOut) {
      try {
        const parsed = JSON.parse(rawOut) as { error?: string };
        if (parsed.error) {
          // Parsed successfully — throw with the clean Python-side message
          throw new Error(parsed.error);
        }
      } catch (innerErr) {
        // Re-throw only if this is OUR clean Error, not a JSON SyntaxError
        if (!(innerErr instanceof SyntaxError)) throw innerErr;
      }
    }

    // Fallback: surface stderr snippet or the Node error message
    const stderrSnippet = (execErr.stderr ?? "").trim().slice(0, 300);
    const msg = stderrSnippet || (err instanceof Error ? err.message : String(err));
    throw new Error(`Python process failed for ${normalizedSymbol}: ${msg}`);
  }

  const elapsed = ((Date.now() - startedAt) / 1000).toFixed(1);
  console.info(`[python-runner] Completed in ${elapsed}s for ${normalizedSymbol}`);

  if (stderr?.trim()) {
    console.warn(`[python-runner] stderr (${normalizedSymbol}):`, stderr.slice(0, 500));
  }

  const raw = stdout.trim();
  if (!raw) {
    throw new Error(`Python script produced no output for ${normalizedSymbol}`);
  }

  let report: PythonReport;
  try {
    report = JSON.parse(raw) as PythonReport;
  } catch {
    throw new Error(
      `Failed to parse JSON output for ${normalizedSymbol}: ${raw.slice(0, 200)}`,
    );
  }

  if (report.error) {
    throw new Error(`Pipeline error for ${normalizedSymbol}: ${report.error}`);
  }

  return report;
}
