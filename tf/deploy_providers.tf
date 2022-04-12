###############################################################################
#
# Terraform: Provider declaration. Download and initialise the Azure Resource
#            Manager API provider plugin.
#
###############################################################################

terraform {
    required_version = ">=0.12"

    required_providers {
        azurerm = {
            source = "hashicorp/azurerm"
            version = "~>2.0"
        }
    }
}

###############################################################################
#
# Azure: Add account and subscription information
#
###############################################################################

provider "azurerm" {
  features {}
  subscription_id	  = "${var.subscription_id}"
  tenant_id			    = "${var.tenant_id}"
}

###############################################################################
#
# Azure: Create resource group
#
###############################################################################

resource "azurerm_resource_group" "rg" {
  name              = "${var.resource_group_name}"
  location          = "${var.resource_group_location}"

  tags = {
    name            = "${var.project-name}"
    investigators   = "${var.project-investigators}"
    contributors    = "${var.project-contributors}"
  }
}