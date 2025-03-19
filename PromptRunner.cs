using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using LLama; // Main namespace for LLamaSharp
using LLama.Common; // For common types like InferenceParams
using Microsoft.Extensions.Logging;

namespace LlamaSharpLoader
{
    public class PromptRunner
    {
        public static async Task Main(string[] args)
        {
            if (args.Length < 1)
            {
                Console.WriteLine("Usage: PromptRunner <modelPath>");
                return;
            }

            string modelPath = args[0];

            // Load the model using LLamaWeights
            var parameters = new ModelParams(modelPath);
            var model = LLamaWeights.LoadFromFile(parameters);

            // Logger setup
            using var loggerFactory = LoggerFactory.Create(builder => builder.AddConsole());
            var logger = loggerFactory.CreateLogger<PromptRunner>();

            // Initialize the context using the loaded model
            var context = new LLamaContext(model, parameters);

            // Create an executor for inference
            var executor = new InstructExecutor(context);

            Console.WriteLine("Model loaded. Type your prompts below. Type 'exit' to quit.");

            // Interactive loop
            while (true)
            {
                Console.Write("\nYou: ");
                string prompt = Console.ReadLine() ?? string.Empty;
                 
                // Exit condition
                if (prompt.ToLower() == "exit")
                {
                    Console.WriteLine("Exiting...");
                    break;
                }

                // Create inference parameters
                var inferenceParams = new InferenceParams
                {
                    MaxTokens = 128000, // Increase the token limit to allow for a longer response
                    AntiPrompts = new List<string>() // No anti-prompts to stop generation prematurely
                };

                // Generate response
                Console.WriteLine("Model:");
                await foreach (var response in executor.InferAsync(prompt, inferenceParams))
                {
                    Console.Write(response); // Output the response
                }
                Console.WriteLine();
            }

            // Cleanup
            Console.WriteLine("Model session completed and deallocated.");
        }
    }
}
