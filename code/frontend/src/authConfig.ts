import { Configuration, PopupRequest } from "@azure/msal-browser";

// Config object to be passed to Msal on creation
export const msalConfig: Configuration = {
    auth: {
      clientId: "a940f5ca-6e24-4b7c-89be-e4621cd553bb",
      authority: "https://login.microsoftonline.com/abdf0bc1-a86f-4288-85a1-0183f205e81f"
    },
    system: {
      allowNativeBroker: false // Disables WAM Broker
    }
};

// Add here scopes for id token to be used at MS Identity Platform endpoints.
export const loginRequest: PopupRequest = {
    scopes: ["User.Read"]
};

// Add here the endpoints for MS Graph API services you would like to use.
export const graphConfig = {
    graphMeEndpoint: "https://graph.microsoft.com/v1.0/me"
};

export const appRoles = ['TaskUser', 'TaskAdmin']; // '*' - all user have access

// export const groupIds = ["1aef6fd4-0eff-4d0d-a2be-6da05f080467"]; // '*' - all user have access
export const groupIds = ["7829d37c-55f9-43ea-8f68-7458e3875f81"];
// export const groupIds = ["*"]