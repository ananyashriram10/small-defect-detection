"use client";

import { useParams } from "next/navigation";
import Link from "next/link";
import { useState } from "react";
import data from "@/lib/data.json";

const SIZE_COLORS = {
  small:  { text: "text-emerald-400", bg: "bg-emerald-500/10", border: "border-emerald-500/30", bar: "#10b981", active: "bg-emerald-500/20 border-emerald-400 text-emerald-300" },
  medium: { text: "text-amber-400",   bg: "bg-amber-500/10",   border: "border-amber-500/30",   bar: "#f59e0b", active: "bg-amber-500/20 border-amber-400 text-amber-300" },
  large:  { text: "text-rose-400",    bg: "bg-rose-500/10",    border: "border-rose-500/30",     bar: "#f43f5e", active: "bg-rose-500/20 border-rose-400 text-rose-300" },
};

const DATASET_ICONS: Record<string, string> = {
  "GC10-DET": "⚙️", "MTD": "🧲", "DAGM": "🔬",
  "KolektorSDD2": "⚡", "MPDD": "🔩", "Severstal": "🏗️", "VisA": "🔍",
};

export default function DatasetPage() {
  const params = useParams();
  const name = decodeURIComponent(params.dataset as string);
  const ds = (data.datasets as any)[name];
  const [activeSize, setActiveSize] = useState<"small" | "medium" | "large">("small");
  const [failedImgs, setFailedImgs] = useState<Set<string>>(new Set());

  if (!ds) return (
    <main className="min-h-screen bg-[#0a0a0a] text-white flex items-center justify-center">
      <div className="text-white/40">Dataset not found</div>
    </main>
  );

  const samples: string[] = ds.samples[activeSize] ?? [];

  return (
    <main className="min-h-screen bg-[#0a0a0a] text-white">
      {/* Header */}
      <div className="border-b border-white/5 bg-[#0a0a0a]/80 backdrop-blur sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center gap-3">
          <Link href="/" className="text-white/40 hover:text-white transition-colors text-sm">← Back</Link>
          <span className="text-white/20">/</span>
          <span className="text-sm font-medium">{name}</span>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-6 py-10 space-y-10">

        {/* Dataset header */}
        <div className="flex items-start gap-4">
          <span className="text-4xl">{DATASET_ICONS[name] ?? "📁"}</span>
          <div className="space-y-1">
            <h1 className="text-3xl font-bold">{name}</h1>
            <p className="text-white/50">{ds.surface}</p>
            <div className="flex flex-wrap gap-1.5 pt-2">
              {ds.defects.map((d: string) => (
                <span key={d} className="text-xs bg-white/5 text-white/50 rounded-md px-2 py-0.5">{d}</span>
              ))}
            </div>
          </div>
        </div>

        {/* Stats row */}
        <div className="grid grid-cols-4 gap-3">
          {[
            { label: "Total", value: ds.total, color: "text-white" },
            { label: "Small",  value: ds.counts.small,  color: "text-emerald-400" },
            { label: "Medium", value: ds.counts.medium, color: "text-amber-400" },
            { label: "Large",  value: ds.counts.large,  color: "text-rose-400" },
          ].map((s) => (
            <div key={s.label} className="rounded-xl border border-white/8 bg-white/[0.03] p-4 space-y-1">
              <p className="text-xs text-white/40 uppercase tracking-widest">{s.label}</p>
              <p className={`text-2xl font-bold tabular-nums ${s.color}`}>{s.value.toLocaleString()}</p>
            </div>
          ))}
        </div>

        {/* Size tabs */}
        <div className="space-y-6">
          <div className="flex gap-2">
            {(["small", "medium", "large"] as const).map((size) => {
              const c = SIZE_COLORS[size];
              const isActive = activeSize === size;
              return (
                <button key={size} onClick={() => setActiveSize(size)}
                  className={`px-4 py-2 rounded-lg border text-sm font-medium transition-all ${
                    isActive ? c.active : `border-white/10 text-white/40 hover:text-white/70 hover:border-white/20`
                  }`}>
                  {size.charAt(0).toUpperCase() + size.slice(1)}
                  <span className="ml-2 opacity-60">{ds.counts[size].toLocaleString()}</span>
                </button>
              );
            })}
          </div>

          {/* Image grid */}
          {samples.length === 0 ? (
            <div className="text-white/30 text-sm py-12 text-center">No images in this size category</div>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
              {samples.filter(url => !failedImgs.has(url)).map((url, i) => (
                <div key={i} className="group rounded-xl overflow-hidden border border-white/8 bg-white/[0.03] aspect-square relative">
                  <img
                    src={url}
                    alt={`${name} ${activeSize} defect ${i + 1}`}
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                    onError={() => setFailedImgs(prev => new Set([...prev, url]))}
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                  <div className={`absolute bottom-2 left-2 text-xs px-2 py-0.5 rounded-md ${SIZE_COLORS[activeSize].bg} ${SIZE_COLORS[activeSize].text} border ${SIZE_COLORS[activeSize].border} opacity-0 group-hover:opacity-100 transition-opacity`}>
                    {activeSize}
                  </div>
                </div>
              ))}
            </div>
          )}

          <p className="text-xs text-white/25">
            Showing {Math.min(samples.length, 12)} sample images · Full dataset on{" "}
            <a href={`https://huggingface.co/datasets/ananya098/SmallDefectDataseet`}
              target="_blank" rel="noopener noreferrer"
              className="text-white/50 hover:text-white underline underline-offset-2">HuggingFace</a>
          </p>
        </div>
      </div>
    </main>
  );
}
