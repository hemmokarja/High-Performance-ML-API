"""
Utility script for managing API keys.

Usage:
    python -m gateway.utils.key_management generate
    python -m gateway.utils.key_management list
    python -m gateway.utils.key_management revoke <key>
"""

import argparse
import sys
from gateway.auth.api_key_db import ApiKeyDB


def generate_key(args):
    """Generate a new API key"""
    api_key = ApiKeyDB.generate_key(prefix=args.prefix)
    
    print("\n" + "="*60)
    print("Generated new API key:")
    print(f"  Key: {api_key}")
    print(f"  Prefix: {args.prefix}")
    print("\nTo use this key:")
    print(f"  export TEST_API_KEY='{api_key}'")
    print("\nOr in your requests:")
    print(f"  Authorization: Bearer {api_key}")
    print("="*60 + "\n")
    
    return api_key


def list_keys(args):
    """List all API keys (for demonstration)"""
    print("\nNote: This is a demo function.")
    print("In production, keys should be listed from your database.\n")
    print("Example key format:")
    print("  sk_live_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789")
    print()


def revoke_key(args):
    """Revoke an API key (for demonstration)"""
    print(f"\nNote: This is a demo function.")
    print(f"In production, call api_key_db.revoke_key('{args.key}')\n")


def main():
    parser = argparse.ArgumentParser(
        description="API Key Management Utility"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Generate command
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate a new API key"
    )
    generate_parser.add_argument(
        "--prefix",
        type=str,
        default="sk_live",
        help="Key prefix (e.g., sk_live, sk_test)"
    )
    
    # List command
    list_parser = subparsers.add_parser(
        "list",
        help="List all API keys"
    )
    
    # Revoke command
    revoke_parser = subparsers.add_parser(
        "revoke",
        help="Revoke an API key"
    )
    revoke_parser.add_argument(
        "key",
        type=str,
        help="API key to revoke"
    )
    
    args = parser.parse_args()
    
    if args.command == "generate":
        generate_key(args)
    elif args.command == "revoke":
        revoke_key(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()