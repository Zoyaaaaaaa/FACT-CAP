import { inngest } from "./client";

const helloWorld = inngest.createFunction(
    { id: "hello-world" },
    { event: "test/hello.world" },
    async ({ event, step }) => {
        await step.sleep("wait-a-moment", "1s");
        return { message: `Hello ${event.data.email}!` };
    },
);

export const inngestFunctions = [helloWorld];

// New function for image prediction
const predictImage = inngest.createFunction(
    { id: "predict-image" },
    { event: "image/predict" },
    async ({ event, step }) => {
        const { imagePath } = event.data;

        const result = await step.run("run-prediction-script", async () => {
            const { spawn } = await import("child_process");
            const path = await import("path");

            return new Promise((resolve, reject) => {
                const scriptPath = path.resolve(process.cwd(), "../../scripts/predict.py");
                // Adjust path relative to where nextjs runs (usually project root: frontend)
                const absoluteScriptPath = path.join(process.cwd(), "../scripts", "predict.py");
                const modelPath = path.join(process.cwd(), "../deepfake_cnn_cpuv1.pth");

                const pythonProcess = spawn("python", [absoluteScriptPath, imagePath, modelPath]);

                let output = "";
                let errorOutput = "";

                pythonProcess.stdout.on("data", (data) => {
                    output += data.toString();
                });

                pythonProcess.stderr.on("data", (data) => {
                    errorOutput += data.toString();
                });

                pythonProcess.on("close", (code) => {
                    if (code !== 0) {
                        reject(new Error(`Script failed with code ${code}: ${errorOutput}`));
                    } else {
                        try {
                            resolve(JSON.parse(output));
                        } catch (e) {
                            reject(new Error("Failed to parse Python output"));
                        }
                    }
                });
            });
        });

        return { result };
    }
);

inngestFunctions.push(predictImage);
