"use client";

import { useState } from "react";
import Image from "next/image";


const Prediction = () => {
    const [preview, setPreview] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [message, setMessage] = useState("");

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            setPreview(URL.createObjectURL(file));
        }
    };

    const handleSubmit = async (formData: FormData) => {
        setLoading(true);
        setMessage("");
        try {
            const res = await fetch("/api/predict", { method: "POST", body: formData });
            if (!res.ok) {
                const text = await res.text();
                throw new Error(text || "Request failed");
            }
            const result = await res.json();
            setMessage(`Success: ${result.message}. Check Inngest dashboard for results.`);
        } catch (error) {
            console.error(error);
            setMessage("Error starting prediction.");
        } finally {
            setLoading(false);
        }
    };

    const onSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        const fd = new FormData(e.currentTarget);
        await handleSubmit(fd);
    };

    return (
        <div className="flex flex-col items-center gap-6 p-8 border rounded-xl shadow-lg bg-white dark:bg-zinc-900 max-w-md w-full">
            <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-500 to-violet-500 bg-clip-text text-transparent">
                Deepfake Detector
            </h1>

            <p className="text-zinc-500 text-center text-sm">
                Upload an image to analyze authenticity
            </p>

            <form onSubmit={onSubmit} className="flex flex-col items-center w-full gap-4">
                <div className="w-full relative group">
                    <input
                        type="file"
                        name="image"
                        accept="image/*"
                        onChange={handleFileChange}
                        className="w-full text-sm text-zinc-500
                        file:mr-4 file:py-2 file:px-4
                        file:rounded-full file:border-0
                        file:text-sm file:font-semibold
                        file:bg-violet-50 file:text-violet-700
                        hover:file:bg-violet-100 cursor-pointer"
                        required
                    />
                </div>

                {preview && (
                    <div className="relative w-full h-48 rounded-lg overflow-hidden border border-zinc-200">
                        <Image
                            src={preview}
                            alt="Preview"
                            fill
                            className="object-cover"
                        />
                    </div>
                )}

                <button
                    type="submit"
                    disabled={loading}
                    className="w-full py-2.5 rounded-lg bg-black dark:bg-white text-white dark:text-black font-medium hover:opacity-90 disabled:opacity-50 transition-opacity"
                >
                    {loading ? "Analyzing..." : "Analyze Image"}
                </button>
            </form>

            {message && (
                <div className="p-3 bg-zinc-50 dark:bg-zinc-800 rounded text-sm text-center">
                    {message}
                </div>
            )}
        </div>
    );
}

export default Prediction;
