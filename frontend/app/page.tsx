import Link from "next/link";
import data from "@/lib/data.json";

const SIZE_COLORS = {
  small:  { text: "text-emerald-400", bar: "#10b981" },
  medium: { text: "text-amber-400",   bar: "#f59e0b" },
  large:  { text: "text-rose-400",    bar: "#f43f5e" },
};

const DATASET_ICONS: Record<string, string> = {
  "GC10-DET":     "⚙️",
  "MTD":          "🧲",
  "DAGM":         "🔬",
  "KolektorSDD2": "⚡",
  "MPDD":         "🔩",
  "Severstal":    "🏗️",
  "VisA":         "🔍",
};

export default function Home() {
  const { totals, datasets } = data;

  return (
    <main className="min-h-screen bg-[#0a0a0a] text-white">
      {/* Header */}
      <div className="border-b border-white/5 bg-[#0a0a0a]/80 backdrop-blur sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center gap-3">
          <span className="text-xl">🔎</span>
          <span className="font-semibold tracking-tight">SmallDefect</span>
          <span className="text-white/30 text-sm ml-auto">Industrial Defect Dataset Collection</span>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-6 py-12 space-y-14">

        {/* Hero */}
        <div className="space-y-3">
          <h1 className="text-4xl font-bold tracking-tight">
            Small Defect Detection
            <span className="text-white/30 font-normal"> · Dataset Hub</span>
          </h1>
          <p className="text-white/50 max-w-xl">
            Preprocessed industrial defect images across 7 datasets, categorised by defect size relative to image area.
          </p>
        </div>

        {/* Grand total stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { label: "Total Images",  value: totals.total,  color: "text-white" },
            { label: "Small  (<1%)",  value: totals.small,  color: "text-emerald-400" },
            { label: "Medium (1–5%)", value: totals.medium, color: "text-amber-400" },
            { label: "Large  (≥5%)",  value: totals.large,  color: "text-rose-400" },
          ].map((s) => (
            <div key={s.label} className="rounded-xl border border-white/8 bg-white/[0.03] p-5 space-y-1">
              <p className="text-xs text-white/40 uppercase tracking-widest">{s.label}</p>
              <p className={`text-3xl font-bold tabular-nums ${s.color}`}>{s.value.toLocaleString()}</p>
            </div>
          ))}
        </div>

        {/* Dataset grid */}
        <div className="space-y-4">
          <h2 className="text-sm font-medium text-white/40 uppercase tracking-widest">Datasets</h2>
          <div className="grid md:grid-cols-2 gap-4">
            {Object.entries(datasets).map(([name, ds]: [string, any]) => (
              <Link key={name} href={`/${encodeURIComponent(name)}`}>
                <div className="group rounded-xl border border-white/8 bg-white/[0.03] hover:bg-white/[0.06] hover:border-white/15 transition-all p-6 space-y-4 cursor-pointer">
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-3">
                      <span className="text-2xl">{DATASET_ICONS[name] ?? "📁"}</span>
                      <div>
                        <h3 className="font-semibold group-hover:text-white transition-colors">{name}</h3>
                        <p className="text-xs text-white/40 mt-0.5">{ds.surface}</p>
                      </div>
                    </div>
                    <span className="text-white/20 group-hover:text-white/60 transition-colors">→</span>
                  </div>

                  {/* Defect tags */}
                  <div className="flex flex-wrap gap-1.5">
                    {ds.defects.slice(0, 5).map((d: string) => (
                      <span key={d} className="text-xs bg-white/5 text-white/50 rounded-md px-2 py-0.5">{d}</span>
                    ))}
                    {ds.defects.length > 5 && (
                      <span className="text-xs text-white/25">+{ds.defects.length - 5} more</span>
                    )}
                  </div>

                  {/* Size bars */}
                  <div className="space-y-2">
                    {(["small", "medium", "large"] as const).map((size) => {
                      const count = ds.counts[size];
                      const pct = Math.round((count / ds.total) * 100);
                      return (
                        <div key={size} className="flex items-center gap-3">
                          <span className={`text-xs w-14 ${SIZE_COLORS[size].text}`}>{size}</span>
                          <div className="flex-1 bg-white/5 rounded-full h-1.5 overflow-hidden">
                            <div className="h-1.5 rounded-full opacity-80"
                              style={{ width: `${pct}%`, backgroundColor: SIZE_COLORS[size].bar }} />
                          </div>
                          <span className="text-xs text-white/40 tabular-nums w-12 text-right">{count.toLocaleString()}</span>
                        </div>
                      );
                    })}
                  </div>

                  <p className="text-xs text-white/25">{ds.total.toLocaleString()} total images</p>
                </div>
              </Link>
            ))}
          </div>
        </div>
      </div>
    </main>
  );
}
