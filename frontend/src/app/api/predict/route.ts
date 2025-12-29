export const runtime = 'nodejs';

import { NextResponse } from "next/server";
import { writeFile } from "fs/promises";
import { join } from "path";
import * as OS from "os";
import { inngest } from "../../../inngest/client";

export async function POST(req: Request) {
  try {
    const formData = await req.formData();
    const file = formData.get("image") as File | null;
    if (!file) {
      return NextResponse.json({ error: "No image uploaded" }, { status: 400 });
    }

    const arrayBuffer = await file.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);

    const tempDir = OS.tmpdir();
    const fileName = `${Date.now()}-${file.name.replace(/[^a-zA-Z0-9.]/g, "_")}`;
    const filePath = join(tempDir, fileName);

    await writeFile(filePath, buffer);

    await inngest.send({
      name: "image/predict",
      data: { imagePath: filePath },
    });

    return NextResponse.json({ message: "Prediction started", filePath });
  } catch (err) {
    console.error(err);
    return NextResponse.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
