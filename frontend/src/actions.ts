"use server";

import { inngest } from "./inngest/client";
import { writeFile } from "fs/promises";
import { join } from "path";
import * as OS from "os";

export const predictImage = async (formData: FormData) => {
    const file = formData.get("image") as File;
    if (!file) {
        throw new Error("No image uploaded");
    }

    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);

    // Save to tmp directory
    const tempDir = OS.tmpdir();
    // Sanitize filename or use unique id
    const fileName = `${Date.now()}-${file.name.replace(/[^a-zA-Z0-9.]/g, "_")}`;
    const filePath = join(tempDir, fileName);

    await writeFile(filePath, buffer);

    // Trigger Inngest
    await inngest.send({
        name: "image/predict",
        data: {
            imagePath: filePath,
        },
    });

    return { message: "Prediction started", filePath };
};
