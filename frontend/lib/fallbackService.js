// frontend/lib/fallbackService.js
/**
 * Fallback service for when the main API is unavailable
 */

export class FallbackChatService {
  constructor() {
    this.responses = [
      {
        keywords: ['salut', 'bonjour', 'hello', 'hi', 'salaam'],
        response: "Bonjour ! Je suis votre assistant d'orientation éducative. Même si je suis actuellement en mode hors ligne, je peux vous donner quelques informations générales. Comment puis-je vous aider avec votre orientation ?"
      },
      {
        keywords: ['médecine', 'medicine', 'docteur', 'doctor'],
        response: "🏥 **Études de Médecine au Maroc:**\n\nLes études de médecine sont accessibles après le baccalauréat via un concours national très sélectif. Les principales facultés sont:\n• Faculté de Médecine de Rabat\n• Faculté de Médecine de Casablanca\n• Faculté de Médecine de Fès\n\n📚 **Conditions d'accès:**\n• Baccalauréat Sciences Expérimentales ou Sciences Mathématiques\n• Réussir le concours national\n• Moyenne générale élevée recommandée\n\n⚠️ *Service hors ligne - Pour des informations détaillées, veuillez réessayer plus tard.*"
      },
      {
        keywords: ['ingénieur', 'engineering', 'ensa', 'encg', 'école'],
        response: "🎓 **Écoles d'Ingénieurs au Maroc:**\n\n**ENSA (École Nationale des Sciences Appliquées):**\n• Informatique, Génie Civil, Électrique\n• Accès via CNC après 2 ans de prépa\n\n**ENCG (École Nationale de Commerce et Gestion):**\n• Management, Finance, Marketing\n• Accès direct après bac ou via concours\n\n**Autres écoles publiques:**\n• ENSIAS (Informatique)\n• EMI (Ingénierie)\n• INPT (Télécommunications)\n\n⚠️ *Service hors ligne - Informations limitées disponibles.*"
      },
      {
        keywords: ['université', 'fac', 'faculté', 'university'],
        response: "🏫 **Universités Publiques au Maroc:**\n\n**Principales universités:**\n• Université Mohammed V - Rabat\n• Université Hassan II - Casablanca\n• Université Sidi Mohamed Ben Abdellah - Fès\n• Université Cadi Ayyad - Marrakech\n\n**Filières disponibles:**\n• Sciences et Techniques\n• Lettres et Sciences Humaines\n• Sciences Économiques et Gestion\n• Droit\n• Sciences de l'Éducation\n\n💡 *Pour des informations précises sur les programmes et inscriptions, contactez directement les universités.*"
      },
      {
        keywords: ['bourse', 'scholarship', 'aide', 'financement'],
        response: "💰 **Bourses et Aides Financières:**\n\n**Bourses nationales:**\n• Bourse du mérite académique\n• Aide sociale pour familles modestes\n• Bourse d'excellence\n\n**Bourses internationales:**\n• Programme Erasmus+ (Europe)\n• Bourses françaises (Campus France)\n• Fulbright (États-Unis)\n\n📝 **Démarches:**\n• Dossier de candidature\n• Justificatifs de revenus\n• Relevés de notes\n\n⚠️ *Informations générales - Consultez les sites officiels pour les détails.*"
      },
      {
        keywords: ['privé', 'private', 'école privée'],
        response: "🏢 **Établissements Privés au Maroc:**\n\n**Écoles d'ingénieurs privées:**\n• EMSI, SUPTECH, HEM\n• Programmes internationaux\n• Partenariats avec universités étrangères\n\n**Universités privées:**\n• Université Internationale de Rabat\n• Université Mundiapolis\n• Al Akhawayn University\n\n💳 **Considérations:**\n• Frais de scolarité plus élevés\n• Programmes souvent en français/anglais\n• Diplômes reconnus par l'État\n\n⚠️ *Service hors ligne - Vérifiez la reconnaissance des diplômes.*"
      }
    ]

    this.defaultResponse = "❌ **Service temporairement indisponible**\n\nJe ne peux pas accéder à ma base de connaissances complète en ce moment. Voici quelques suggestions :\n\n🔄 **Réessayez dans quelques minutes**\n📞 **Contactez directement les établissements**\n🌐 **Consultez les sites officiels des universités**\n\nPour une assistance complète, attendez que le service soit rétabli."
  }

  /**
   * Generate a fallback response based on keywords
   */
  generateResponse(message) {
    const lowerMessage = message.toLowerCase()
    
    // Find matching response based on keywords
    for (const responseData of this.responses) {
      if (responseData.keywords.some(keyword => lowerMessage.includes(keyword))) {
        return {
          response: responseData.response,
          sources_count: 0,
          confidence: "Low",
          avg_score: 0,
          top_score: 0,
          timestamp: new Date().toISOString(),
          isFallback: true
        }
      }
    }

    // Return default response if no keywords match
    return {
      response: this.defaultResponse,
      sources_count: 0,
      confidence: "Low", 
      avg_score: 0,
      top_score: 0,
      timestamp: new Date().toISOString(),
      isFallback: true
    }
  }

  /**
   * Get example questions for offline mode
   */
  getExamples() {
    return {
      examples: [
        "Bonjour, j'ai besoin d'aide",
        "Comment étudier la médecine ?",
        "Quelles sont les écoles d'ingénieurs ?",
        "Bourses disponibles pour étudiants",
        "Universités publiques au Maroc",
        "Écoles privées reconnues"
      ]
    }
  }
}

// Create singleton instance
export const fallbackService = new FallbackChatService()