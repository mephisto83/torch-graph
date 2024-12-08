type ParameterType = "number" | "boolean" | "string";

export interface Parameter {
    name: string;
    type: ParameterType;
    default: any;
}

export interface PytorchModuleConfig {
    name: string;
    nodeTypes: { [key: string]: string };
    layerParameters: { [key: string]: Parameter[] };
    layerCategories: { [key: string]: string };
    layerToClassMap: { [key: string]: string };
}

export function validateConfig(config: any): { valid: boolean; errors: string[] } {
    const errors: string[] = [];

    // Helper function to check if a value is an object
    const isObject = (obj: any) => obj !== null && typeof obj === "object" && !Array.isArray(obj);

    if (!isObject(config)) {
        errors.push("Config should be an object.");
        return { valid: false, errors };
    }

    const requiredKeys = ["nodeTypes", "layerParameters", "layerCategories", "layerToClassMap"];

    for (const key of requiredKeys) {
        if (!(key in config)) {
            errors.push(`Missing required key: ${key}`);
        } else if (!isObject(config[key])) {
            errors.push(`Key '${key}' should be an object.`);
        }
    }

    if (errors.length > 0) {
        return { valid: false, errors };
    }

    const { nodeTypes, layerParameters, layerCategories, layerToClassMap } = config as PytorchModuleConfig;

    // Validate nodeTypes
    for (const [key, value] of Object.entries(nodeTypes)) {
        if (typeof key !== "string") {
            errors.push(`nodeTypes key '${key}' is not a string.`);
        }
        if (typeof value !== "string") {
            errors.push(`nodeTypes['${key}'] should be a string.`);
        }
    }

    // Validate layerParameters
    for (const [layer, params] of Object.entries(layerParameters)) {
        if (!Array.isArray(params)) {
            errors.push(`layerParameters['${layer}'] should be an array.`);
            continue;
        }

        params.forEach((param, index) => {
            if (!isObject(param)) {
                errors.push(`layerParameters['${layer}'][${index}] should be an object.`);
                return;
            }

            const { name, type, default: defaultValue } = param;

            if (typeof name !== "string") {
                errors.push(`layerParameters['${layer}'][${index}].name should be a string.`);
            }

            if (!["number", "boolean", "string"].includes(type)) {
                errors.push(
                    `layerParameters['${layer}'][${index}].type should be one of 'number', 'boolean', 'string'.`
                );
            }

            // Check if default value matches the type
            if (type === "number" && typeof defaultValue !== "number") {
                errors.push(
                    `layerParameters['${layer}'][${index}].default should be a number.`
                );
            } else if (type === "boolean" && typeof defaultValue !== "boolean") {
                errors.push(
                    `layerParameters['${layer}'][${index}].default should be a boolean.`
                );
            } else if (type === "string" && typeof defaultValue !== "string") {
                errors.push(
                    `layerParameters['${layer}'][${index}].default should be a string.`
                );
            }
        });
    }

    // Validate layerCategories
    for (const [category, layer] of Object.entries(layerCategories)) {
        if (typeof category !== "string") {
            errors.push(`layerCategories key '${category}' is not a string.`);
        }
        if (typeof layer !== "string") {
            errors.push(`layerCategories['${category}'] should be a string.`);
        }
    }

    // Validate layerToClassMap
    for (const [layer, classMap] of Object.entries(layerToClassMap)) {
        if (typeof layer !== "string") {
            errors.push(`layerToClassMap key '${layer}' is not a string.`);
        }
        if (typeof classMap !== "string") {
            errors.push(`layerToClassMap['${layer}'] should be a string.`);
        }
    }

    return { valid: errors.length === 0, errors };
}