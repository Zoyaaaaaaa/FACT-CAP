import { inngest } from "../inngest/client";
import Prediction from "../components/Prediction";
export default function Home() {
  const triggerFunction = async () => {
    "use server";
    await inngest.send({
      name: "test/hello.world",
      data: {
        email: "test@example.com",
      },
    });
  };

  return (
    <div className="flex min-h-screen flex-col gap-4 items-center justify-center bg-zinc-50 font-sans dark:bg-black">
      <div>hiuiii!!!!!!!!!!!</div>
      <Prediction />
      <form action={triggerFunction}>
        <button
          type="submit"
          className="rounded bg-blue-500 px-4 py-2 font-bold text-white hover:bg-blue-700"
        >
          Trigger Inngest Function
        </button>
      </form>
    </div>
  );
}
