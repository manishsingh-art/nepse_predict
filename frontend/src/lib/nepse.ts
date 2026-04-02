// Service layer for NEPSE company data
// Primary source: merolagani.com autosuggest API (full list ~700 entries, filtered to equity only)
// Fallback:       static seed list

const MEROLAGANI_API =
  "https://www.merolagani.com/handlers/AutoSuggestHandler.ashx?type=Company";

// ── In-memory cache ───────────────────────────────────────────────────────────
const memCache = new Map<string, { data: unknown; expiresAt: number }>();

function getCache<T>(key: string): T | null {
  const entry = memCache.get(key);
  if (entry && entry.expiresAt > Date.now()) return entry.data as T;
  return null;
}
function setCache(key: string, data: unknown, ttlMs = 3_600_000): void {
  memCache.set(key, { data, expiresAt: Date.now() + ttlMs });
}
export function invalidateCache(key: string): void {
  memCache.delete(key);
}

// ── Raw shape from merolagani ─────────────────────────────────────────────────
interface MerolaganiEntry {
  l: string; // "ADBL (Agriculture Development Bank Limited)"
  v: string; // numeric id
  d: string; // symbol: "ADBL"
}

// ── Filter: keep only ordinary equity shares ─────────────────────────────────
function isEquityStock(entry: MerolaganiEntry): boolean {
  const label = entry.l.toLowerCase();

  // Promoter shares
  if (label.includes("promoter share") || label.includes("promoter")) return false;

  // Fixed-income instruments: debentures, bonds, rinpatra, bachatpatra
  if (
    label.includes("debenture") ||
    label.includes("rinpatra") ||
    label.includes("bachatpatra") ||
    /^\d/.test(entry.l) // lines starting with a digit are usually "8% XYZ Bond 2083"
  )
    return false;

  // Mutual funds, close-end funds, schemes, yojanas, kosh
  if (
    /\bfund\b/.test(label) ||
    /\bscheme\b/.test(label) ||
    /\byojana\b/.test(label) ||
    /\bkosh\b/.test(label)
  )
    return false;

  // Regulatory / institutional entries that aren't listed equities
  const notEquity = [
    "association",
    "board of nepal",
    "exchange limited",
    "clearing limited",
    "payment solution",
    "securities board",
    "brokers association",
    "bankers association",
    "insurance institute",
    "investment company limited", // some are just holding vehicles
  ];
  if (notEquity.some((kw) => label.includes(kw))) return false;

  // Symbol must be 1-12 uppercase alphanumeric chars only
  if (!/^[A-Z0-9]{1,12}$/.test(entry.d.trim())) return false;

  return true;
}

// ── Parse company name from label ─────────────────────────────────────────────
function parseName(label: string): string {
  const start = label.indexOf("(");
  const end = label.lastIndexOf(")");
  if (start !== -1 && end > start) {
    return label.slice(start + 1, end).trim();
  }
  return label.trim();
}

// ── Infer sector from company name ───────────────────────────────────────────
function inferSector(name: string): string {
  const n = name.toLowerCase();
  if (n.includes("development bank") || n.includes("bikas bank")) return "Development Banks";
  if (
    n.includes("commercial bank") ||
    n.includes("bank limited") ||
    n.includes("bank ltd") ||
    n.includes("bank nepal")
  )
    return "Banking";
  if (n.includes("laghubitta") || n.includes("microfinance") || n.includes("laghubitta"))
    return "Microfinance";
  if (
    n.includes("hydropower") ||
    n.includes("hydro") ||
    n.includes("jalvidhyut") ||
    n.includes("jalbidhyut") ||
    n.includes("jalavidhyut") ||
    n.includes("hydroelectric") ||
    n.includes("power company") ||
    n.includes("urja")
  )
    return "Hydropower";
  if (n.includes("life insurance")) return "Life Insurance";
  if (n.includes("reinsurance")) return "Reinsurance";
  if (n.includes("insurance")) return "Non-Life Insurance";
  if (n.includes("finance")) return "Finance";
  if (
    n.includes("hotel") ||
    n.includes("resort") ||
    n.includes("hospitality") ||
    n.includes("restaurant")
  )
    return "Hotels";
  if (n.includes("cement") || n.includes("steel") || n.includes("textile") || n.includes("mill"))
    return "Manufacturing";
  if (
    n.includes("capital limited") ||
    n.includes("merchant bank") ||
    n.includes("securities limited") ||
    n.includes("investment limited")
  )
    return "Capital Markets";
  if (n.includes("cable car") || n.includes("tourism")) return "Tourism";
  if (n.includes("telecom") || n.includes("media") || n.includes("television")) return "Telecom & Media";
  if (n.includes("pharma") || n.includes("hospital") || n.includes("healthcare")) return "Healthcare";
  return "Others";
}

// ── fetchAllCompanies ─────────────────────────────────────────────────────────

export interface NepseCompany {
  symbol: string;
  name: string;
  sector: string | null;
  nepseId: number;   // merolagani "v" field — higher value = newer listing
}

export async function fetchAllCompanies(): Promise<NepseCompany[]> {
  const cacheKey = "nepse_companies";
  const cached = getCache<NepseCompany[]>(cacheKey);
  if (cached) return cached;

  try {
    const res = await fetch(MEROLAGANI_API, {
      headers: {
        Accept: "application/json",
        "User-Agent": "Mozilla/5.0 (compatible; NEPSE-Dashboard/1.0)",
        Referer: "https://www.merolagani.com/",
      },
      signal: AbortSignal.timeout(15_000),
      cache: "no-store",
    });

    if (!res.ok) throw new Error(`merolagani responded ${res.status}`);

    const raw: MerolaganiEntry[] = await res.json();

      const companies: NepseCompany[] = raw
      .filter(isEquityStock)
      .map((entry) => {
        const symbol = entry.d.trim().toUpperCase();
        const name = parseName(entry.l);
        return {
          symbol,
          name: name || symbol,
          sector: inferSector(name),
          nepseId: parseInt(entry.v, 10) || 0,
        };
      })
      // De-duplicate by symbol
      .filter(
        (c, idx, arr) => arr.findIndex((x) => x.symbol === c.symbol) === idx,
      );

    if (companies.length > 10) {
      setCache(cacheKey, companies, 3_600_000); // 1 h
      console.info(`[nepse.ts] Fetched ${companies.length} equity companies from merolagani`);
      return companies;
    }
  } catch (err) {
    console.warn("[nepse.ts] fetchAllCompanies failed, falling back to seed list:", err);
  }

  // Fallback static seed list (established stocks — nepseId defaults to 0)
  const seed: NepseCompany[] = NEPSE_SEED_COMPANIES.map((c) => ({
    symbol: c.symbol,
    name: c.name,
    sector: c.sector as string | null,
    nepseId: 0,
  }));
  setCache(cacheKey, seed, 600_000);
  return seed;
}

// ── Static fallback seed list ────────────────────────────────────────────────

export const NEPSE_SEED_COMPANIES: Array<{
  symbol: string;
  name: string;
  sector: string;
}> = [
  // Banking
  { symbol: "NABIL", name: "Nabil Bank Limited", sector: "Banking" },
  { symbol: "NICA", name: "NIC Asia Bank Limited", sector: "Banking" },
  { symbol: "EBL", name: "Everest Bank Limited", sector: "Banking" },
  { symbol: "SCB", name: "Standard Chartered Bank Nepal Limited", sector: "Banking" },
  { symbol: "KBL", name: "Kumari Bank Limited", sector: "Banking" },
  { symbol: "NBL", name: "Nepal Bank Limited", sector: "Banking" },
  { symbol: "SBI", name: "Nepal SBI Bank Limited", sector: "Banking" },
  { symbol: "ADBL", name: "Agricultural Development Bank Limited", sector: "Banking" },
  { symbol: "SANIMA", name: "Sanima Bank Limited", sector: "Banking" },
  { symbol: "PRVU", name: "Prabhu Bank Limited", sector: "Banking" },
  { symbol: "MBL", name: "Machhapuchchhre Bank Limited", sector: "Banking" },
  { symbol: "GBIME", name: "Global IME Bank Limited", sector: "Banking" },
  { symbol: "PCBL", name: "Prime Commercial Bank Limited", sector: "Banking" },
  { symbol: "NMB", name: "NMB Bank Limited", sector: "Banking" },
  { symbol: "SHL", name: "Siddhartha Bank Limited", sector: "Banking" },
  { symbol: "NIBL", name: "Nabil Investment Banking Limited", sector: "Banking" },
  { symbol: "CBL", name: "Civil Bank Limited", sector: "Banking" },
  { symbol: "BOKL", name: "Bank of Kathmandu Limited", sector: "Banking" },
  { symbol: "MEGA", name: "Mega Bank Nepal Limited", sector: "Banking" },
  { symbol: "RBB", name: "Rastriya Banijya Bank Limited", sector: "Banking" },
  { symbol: "NIMB", name: "Nepal Investment Mega Bank Limited", sector: "Banking" },
  { symbol: "LSL", name: "Laxmi Sunrise Bank Limited", sector: "Banking" },
  // Development Banks
  { symbol: "LBBL", name: "Lumbini Bikas Bank Limited", sector: "Development Banks" },
  // Hydropower
  { symbol: "HIDCL", name: "Hydroelectricity Investment and Development Company Limited", sector: "Hydropower" },
  { symbol: "CHCL", name: "Chilime Hydropower Company Limited", sector: "Hydropower" },
  { symbol: "BPCL", name: "Butwal Power Company Limited", sector: "Hydropower" },
  { symbol: "UPPER", name: "Upper Tamakoshi Hydropower Limited", sector: "Hydropower" },
  { symbol: "AKJCL", name: "Ankhu Khola Jalvidhyut Company Limited", sector: "Hydropower" },
  { symbol: "API", name: "Api Power Company Limited", sector: "Hydropower" },
  { symbol: "NGPL", name: "Ngadi Group Power Limited", sector: "Hydropower" },
  { symbol: "RRHP", name: "Rairang Hydropower Development Company Limited", sector: "Hydropower" },
  // Insurance
  { symbol: "NLIC", name: "Nepal Life Insurance Company Limited", sector: "Life Insurance" },
  { symbol: "LICN", name: "Life Insurance Corporation Nepal Limited", sector: "Life Insurance" },
  { symbol: "HGI", name: "Himalayan General Insurance Company Limited", sector: "Non-Life Insurance" },
  { symbol: "PRIN", name: "Prabhu Insurance Limited", sector: "Non-Life Insurance" },
  // Finance & Others
  { symbol: "CIT", name: "Citizen Investment Trust", sector: "Finance" },
  { symbol: "NTC", name: "Nepal Doorsanchar Company Limited", sector: "Telecom & Media" },
  { symbol: "TRH", name: "Taragaon Regency Hotel Limited", sector: "Hotels" },
  { symbol: "SHIVM", name: "Shivam Cements Limited", sector: "Manufacturing" },
  { symbol: "NMBMF", name: "NMB Microfinance Bittiya Sanstha Limited", sector: "Microfinance" },
  { symbol: "SKBBL", name: "Sana Kisan Bikas Bank Limited", sector: "Development Banks" },
];

// ── fetchStockPrice ───────────────────────────────────────────────────────────

export interface LivePrice {
  symbol: string;
  lastPrice: number;
  change: number;
  changePct: number;
  volume: number;
  date: string;
}

export async function fetchStockPrice(symbol: string): Promise<LivePrice | null> {
  const cacheKey = `price_${symbol}`;
  const cached = getCache<LivePrice>(cacheKey);
  if (cached) return cached;

  try {
    const res = await fetch(
      `https://nepalstock.com.np/api/nots/security/${encodeURIComponent(symbol)}`,
      {
        headers: { Accept: "application/json" },
        signal: AbortSignal.timeout(8_000),
      },
    );
    if (!res.ok) return null;
    const json = await res.json();
    const raw = json?.data ?? json;
    const price: LivePrice = {
      symbol,
      lastPrice: Number(raw?.lastTradedPrice ?? raw?.lastPrice ?? 0),
      change: Number(raw?.priceChange ?? raw?.change ?? 0),
      changePct: Number(raw?.percentageChange ?? raw?.changePct ?? 0),
      volume: Number(raw?.totalTradeQuantity ?? raw?.volume ?? 0),
      date: raw?.businessDate ?? raw?.date ?? new Date().toISOString().slice(0, 10),
    };
    setCache(cacheKey, price, 300_000);
    return price;
  } catch {
    return null;
  }
}
